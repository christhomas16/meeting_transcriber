#!/usr/bin/env python3
"""
Meeting Transcriber - Using Hugging Face ASR + Diarization Pipeline
Based on: https://huggingface.co/blog/asr-diarization

Uses an integrated ASR + Diarization pipeline for accurate speaker identification.
"""

import warnings
import os
import sys
import logging
import argparse

# Suppress all informational and warning messages from PyTorch Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Suppress specific warnings from other libraries that are less aggressive
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.pipelines.speaker_verification")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.tasks.segmentation.mixins")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.model")
warnings.filterwarnings("ignore", message=".*`ModelCheckpoint` callback states.*")

import time
import queue
import threading
import numpy as np
import sounddevice as sd
import torch
from datetime import datetime
from dotenv import load_dotenv
# Disable pyannote 4.0 telemetry (anonymous usage metrics sent to otel.pyannote.ai).
# Must be set BEFORE importing pyannote.audio. Keeps the app fully offline; no audio
# or transcript is ever sent regardless, but this stops the metrics ping too.
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
# NOTE: `transformers` is imported lazily inside the legacy model-init path so the
# modern engine (--modern) can run in an environment that doesn't install it.
from pyannote.audio import Pipeline
from pyannote.audio import Model as PyannoteModel
from pyannote.audio import Inference
from scipy.spatial.distance import cdist
import requests
import json

class MeetingTranscriber:
    OLLAMA_URL = "http://localhost:11434"

    # Final summary prompt (run over the transcript, or over chunk summaries for long meetings).
    SUMMARY_PROMPT = """You are an expert meeting summarizer. Provide a concise, easy-to-read summary of the following meeting transcript.

Identify the main topics of discussion and any key decisions or action items. Structure the summary as a brief overview followed by bullet points for the main topics, then a short "Action Items" section if any were mentioned.

Here is the transcript:
---
{transcript}
---
"""

    # Map step prompt: condense one chunk of a long transcript before the final reduce.
    CHUNK_PROMPT = """Summarize this portion of a meeting transcript into a few tight bullet points, preserving any decisions, action items, names, numbers, and dates. Do not add a preamble.

Transcript portion:
---
{transcript}
---
"""

    def __init__(self, model_name="openai/whisper-medium.en", debug=False, ollama_model="llama3.2", live_transcription=False, use_modern=False, meeting_mode=False, max_minutes=None, input_device=None, summary_only=False):
        """
        Initialize the meeting transcriber.

        Args:
            model_name (str): Whisper model to use (legacy engine only)
            debug (bool): Whether to enable debug mode
            ollama_model (str): The name of the Ollama model to use for summarization
            live_transcription (bool): Whether to enable live transcription preview
            use_modern (bool): Use the modern Parakeet-MLX + pyannote-community-1 engine
            meeting_mode (bool): Passive capture: one continuous recording until Ctrl+C,
                                 then a single transcript + summary (no 5-minute chunking)
            max_minutes (float): Safety auto-stop duration. Defaults to 180 in meeting
                                 mode, 5 otherwise.
        """
        self.debug = debug
        self.whisper_model_name = model_name
        self.ollama_model = ollama_model
        self.use_modern = use_modern
        self.meeting_mode = meeting_mode
        self.summary_only = summary_only  # re-summarize a transcript; skip ASR/diarization models

        # In passive meeting capture we record the whole meeting and process once;
        # live preview is unnecessary and just competes for the GPU during the call.
        if meeting_mode and live_transcription:
            print("ℹ️  Live transcription is disabled in meeting mode (passive capture).")
            live_transcription = False
        # Live preview is only wired into the legacy ASR pipeline; the modern engine
        # processes the full buffer at the end, so disable live preview there.
        if use_modern and live_transcription:
            print("ℹ️  Live transcription is not supported with the modern engine; disabling it.")
            live_transcription = False
        self.live_transcription = live_transcription
        self.session_start_time = datetime.now()  # Track when this session started

        # Load environment variables and check for token (not needed for re-summary,
        # which uses only Ollama).
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token and not summary_only:
            print("\nError: Hugging Face token not found. Please see README.md for setup instructions.")
            sys.exit(1)

        # Audio parameters
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        self.input_device = input_device  # None = system default input
        self.is_recording = False
        self.is_stopping = False

        # Main buffer for final, high-quality processing.
        # Holds a list of numpy chunks (not individual samples); concatenated once
        # at processing time. audio_buffer_samples tracks the total sample count so
        # we never have to call len() on a multi-million-element Python list.
        self.audio_buffer = []
        self.audio_buffer_samples = 0
        if max_minutes is None:
            max_minutes = 180 if meeting_mode else 5  # generous safety cap for meetings
        self.max_duration_seconds = int(max_minutes * 60)
        self.max_buffer_size = int(self.sample_rate * self.max_duration_seconds)

        # Real-time processing components (only used if live_transcription is True)
        self.realtime_audio_queue = queue.Queue()
        self.realtime_chunk_duration = 5  # Process in 5-second chunks for live feedback
        self.realtime_overlap_duration = 1  # 1-second overlap for better context
        self.realtime_buffer = []
        self.realtime_buffer_size = int(self.sample_rate * self.realtime_chunk_duration)
        self.realtime_overlap_size = int(self.sample_rate * self.realtime_overlap_duration)
        self.last_realtime_text = ""
        self.vad_threshold = 0.001  # Lowered energy threshold for more sensitive VAD

        # Transcript storage
        self.text_buffer = []

        # If the transcript exceeds this many characters, summarize it via map-reduce
        # (summarize chunks, then summarize the chunk summaries) so long meetings
        # don't silently overflow a small model's context window.
        self.summary_char_threshold = 14000
        
        # Speaker identification components
        self.person_counter = 1
        self.speaker_voiceprints = {}

        # --- Model Initialization ---
        # Re-summary mode only talks to Ollama; skip loading any ASR/diarization models.
        if self.summary_only:
            self.device_info = "cpu"
            self.asr_label = "(re-summary of existing transcript)"
            self.diar_label = "(re-summary of existing transcript)"
            self.output_filename = None
            print(f"Re-summary mode: skipping model load, summarizing with Ollama ({self.ollama_model}).\n")
            return

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

        if self.use_modern:
            print("Initializing modern engine (Parakeet-MLX + pyannote community-1)...")
            try:
                from modern_transcribe import ModernEngine
                self.engine = ModernEngine(hf_token=hf_token, device=device, debug=debug)
                self.device_info = device
                self.asr_label = "mlx-community/parakeet-tdt-0.6b-v3 (parakeet-mlx)"
                self.diar_label = "pyannote/speaker-diarization-community-1"
                print(f"Modern engine loaded successfully on device: {self.device_info}\n")
            except Exception as e:
                print(f"\nError during modern engine initialization: {e}")
                sys.exit(1)
            self.output_filename = None
            return

        print("Initializing ASR + Diarization pipeline...")
        try:
            # Initialize ASR pipeline (lazy transformers import keeps it out of the modern env)
            from transformers import pipeline
            torch_dtype = torch.float32

            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch_dtype,
                device=device,
                return_timestamps=True,
                generate_kwargs={"max_new_tokens": 448}
            )

            # Initialize Diarization pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            # Initialize Speaker Embedding model for voiceprinting
            self.embedding_model = PyannoteModel.from_pretrained(
                "pyannote/embedding",
                use_auth_token=hf_token
            )

            if device == "mps":
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("mps"))
                self.embedding_model.to(torch.device("mps"))
            elif device == "cuda":
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
                self.embedding_model.to(torch.device("cuda"))

            # Build the embedding Inference once and reuse it for every segment.
            # Previously this was reconstructed on every call to get_embedding(),
            # which is run once per diarization turn -- a significant waste.
            self.embedding_inference = Inference(self.embedding_model, window="whole")

            self.device_info = device
            self.asr_label = self.whisper_model_name
            self.diar_label = "pyannote/speaker-diarization-3.1"
            print(f"Models loaded successfully on device: {self.device_info}\n")
        except Exception as e:
            print(f"\nError during model initialization: {e}")
            sys.exit(1)

        # File output
        self.output_filename = None

    def process_audio_buffer(self):
        """Process the entire audio buffer with ASR + Diarization."""
        if not self.audio_buffer:
            print("No audio recorded.")
            return

        print(f"\nProcessing {self.audio_buffer_samples / self.sample_rate:.1f}s of recorded audio...")
        audio_data = np.concatenate(self.audio_buffer).astype(np.float32)

        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data /= np.max(np.abs(audio_data))

        # SAFETY: persist the raw recording to disk BEFORE transcription, so a
        # processing failure can never lose the audio. It can be re-transcribed
        # later with --from-audio <path>.
        self._save_recording_wav(audio_data)

        try:
            if self.use_modern:
                # Modern engine: Parakeet-MLX ASR + pyannote community-1 diarization
                # + sentence-level speaker alignment, all in-memory.
                if self.debug: print("🚀 Running modern engine (Parakeet + community-1)...")
                segments = self.engine.transcribe_array(audio_data, self.sample_rate)
            else:
                # Feed audio to pyannote in-memory (waveform, sample_rate) instead of
                # writing/reading a temp WAV file. pyannote's Audio supports this dict
                # protocol for both the diarization pipeline and Inference.crop, so we
                # avoid the disk round-trip entirely.
                waveform = torch.from_numpy(audio_data).unsqueeze(0)  # shape: (channel, time)
                self.audio_source = {"waveform": waveform, "sample_rate": self.sample_rate}

                # Run diarization
                if self.debug: print("🎯 Running diarization...")
                diarization = self.diarization_pipeline(self.audio_source)

                # Run ASR with chunking for long-form audio
                if self.debug: print("🎤 Running ASR transcription...")
                asr_result = self.asr_pipeline(
                    audio_data,
                    return_timestamps=True,
                    chunk_length_s=30,
                    stride_length_s=5
                )

                # Combine ASR and diarization results
                segments = self.align_asr_with_diarization(asr_result, diarization)

            # Display and store results
            if segments:
                print("\n" + "="*25 + " FINAL TRANSCRIPT " + "="*25)
                for segment in segments:
                    start_time = segment.get('start_time', 0)
                    print(f"[{start_time:0>7.2f}s] 👤 {segment['speaker']}: {segment['text']}")
                    self.text_buffer.append(f"[{start_time:0>7.2f}s] {segment['speaker']}: {segment['text']}")
                print("="*68 + "\n")

        except Exception as e:
            print(f"Error processing audio: {str(e)}", file=sys.stderr)
            if getattr(self, 'last_recording_path', None):
                print(f"💾 Your audio is safe at: {self.last_recording_path}", file=sys.stderr)
                print(f"   Re-transcribe it with:  python meeting_transcriber.py --modern "
                      f"--from-audio \"{self.last_recording_path}\"", file=sys.stderr)

    def process_transcript_file(self, path):
        """Re-summarize an existing saved transcript with the configured Ollama model
        (no ASR/diarization). Writes a fresh summary + transcript file pair."""
        with open(path, "r", encoding="utf-8") as f:
            # Keep only the actual dialogue lines, e.g. "[0012.34s] SPEAKER_00: ..."
            lines = [ln.rstrip("\n") for ln in f if ln.lstrip().startswith("[")]
        if not lines:
            print(f"No transcript lines found in {path} (expected lines starting with '[').")
            return
        self.text_buffer = lines
        print(f"📄 Loaded {len(lines)} transcript lines from {path}")
        print(f"✨ Re-summarizing with {self.ollama_model}...")
        summary = self.generate_summary()
        filename = self.save_meeting_to_file(summary)
        if filename:
            print("\n" + "=" * 68)
            print(f"✅ New summary saved to transcriptions/ folder:")
            print(f"   📄 {filename.split(' and ')[0]}")
            print("=" * 68 + "\n")

    def process_audio_file(self, path):
        """Re-transcribe a previously saved WAV (no microphone). Runs on the main
        thread, which is required for the MLX-based modern engine."""
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        print(f"📂 Re-processing saved audio: {path} ({audio.shape[0]/sr:.1f}s @ {sr}Hz)")
        self.reprocessing = True          # don't re-save; it's already a file
        self.last_recording_path = path
        self.audio_buffer = [audio]
        self.audio_buffer_samples = audio.shape[0]
        self.sample_rate = sr
        self._background_processing()

    def _save_recording_wav(self, audio_data):
        """Persist the raw recording to transcriptions/ before transcription so the
        audio is never lost if processing fails. Sets self.last_recording_path."""
        if getattr(self, 'reprocessing', False):
            return
        try:
            import soundfile as sf
            os.makedirs("transcriptions", exist_ok=True)
            ts = datetime.now().strftime("%m-%d-%y-%H%M")
            path = os.path.join("transcriptions", f"meeting-audio-{ts}.wav")
            counter = 1
            while os.path.exists(path):
                path = os.path.join("transcriptions", f"meeting-audio-{ts}-{counter}.wav")
                counter += 1
            sf.write(path, audio_data, self.sample_rate)
            self.last_recording_path = path
            print(f"💾 Raw recording saved: {path}")
        except Exception as e:
            self.last_recording_path = None
            print(f"⚠️  Could not save raw recording: {e}", file=sys.stderr)

    def get_embedding(self, audio_source, segment):
        """Extracts an embedding for a given audio source and speaker segment."""
        # Segments shorter than a certain duration can cause errors in the embedding model.
        MIN_DURATION = 0.05  # 50 milliseconds, a safe threshold for pyannote/embedding
        if segment.duration < MIN_DURATION:
            if self.debug:
                print(f"⏩ Skipping embedding for very short segment ({segment.duration:.3f}s)", file=sys.stderr)
            return None

        try:
            # Reuse the Inference object built in __init__. The 'crop' method takes
            # the audio source (in-memory dict or path) and the segment to extract.
            embedding = self.embedding_inference.crop(audio_source, segment)
            return embedding
        except Exception as e:
            # The duration check should prevent most errors, but this is a fallback.
            if self.debug:
                print(f"⚠️  Error getting embedding for segment {segment.start:.2f}-{segment.end:.2f}s: {e}", file=sys.stderr)
            return None

    def align_asr_with_diarization(self, asr_result, diarization):
        """Align ASR chunks with speaker diarization using voice embeddings."""
        segments = []
        asr_chunks = asr_result.get("chunks", [])
        if not asr_chunks:
            return segments

        # Single pass over diarization tracks: collect per-speaker embeddings and
        # total durations, and remember every turn for the timeline below. This
        # replaces three separate itertracks() walks plus an O(n^2) duration sum.
        speaker_data = {}  # label -> {'embeddings': [...], 'duration': float}
        turns = []  # list of (turn, speaker_label)
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            data = speaker_data.setdefault(speaker_label, {'embeddings': [], 'duration': 0.0})
            data['duration'] += turn.end - turn.start

            embedding = self.get_embedding(self.audio_source, turn)
            if embedding is not None:
                data['embeddings'].append(embedding)

            turns.append((turn, speaker_label))

        # Create a mapping from chunk speaker labels to consistent person names
        chunk_speaker_map = {}
        for speaker_label, data in speaker_data.items():
            if not data['embeddings']:
                continue
            avg_embedding = np.mean(data['embeddings'], axis=0)
            person_name = self.get_consistent_person_name_by_voice(avg_embedding, data['duration'])
            chunk_speaker_map[speaker_label] = person_name

        # Create a speaker timeline using the consistent names
        speaker_timeline = []
        for turn, speaker_label in turns:
            if speaker_label in chunk_speaker_map:
                speaker_timeline.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': chunk_speaker_map[speaker_label]
                })

        if self.debug:
            print(f"📊 Found {len(speaker_timeline)} speaker segments with consistent names")
            print(f"📝 Found {len(asr_chunks)} ASR chunks")

        # Assign speakers to ASR chunks
        for chunk in asr_chunks:
            chunk_start = chunk['timestamp'][0] if chunk['timestamp'][0] is not None else 0
            chunk_end = chunk['timestamp'][1] if chunk['timestamp'][1] is not None else chunk_start + 1
            chunk_text = chunk['text'].strip()

            if not chunk_text:
                continue

            best_speaker = self.find_best_speaker(chunk_start, chunk_end, speaker_timeline)

            if best_speaker:
                segments.append({
                    'speaker': best_speaker,
                    'text': chunk_text,
                    'start_time': chunk_start,
                    'end_time': chunk_end
                })
                if self.debug:
                    print(f"🎯 [{chunk_start:.1f}s-{chunk_end:.1f}s] {best_speaker}: {chunk_text[:50]}...")

        return segments

    def find_best_speaker(self, chunk_start, chunk_end, speaker_timeline):
        """Find the speaker with the best overlap for a given time chunk"""
        best_speaker = None
        best_overlap = 0

        chunk_duration = chunk_end - chunk_start

        for speaker_segment in speaker_timeline:
            # Calculate overlap
            overlap_start = max(chunk_start, speaker_segment['start'])
            overlap_end = min(chunk_end, speaker_segment['end'])
            overlap_duration = max(0, overlap_end - overlap_start)

            # Calculate overlap ratio
            overlap_ratio = overlap_duration / chunk_duration if chunk_duration > 0 else 0

            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_speaker = speaker_segment['speaker']

        return best_speaker if best_overlap > 0.1 else None  # Require at least 10% overlap

    def get_consistent_person_name_by_voice(self, embedding, duration, 
                                            strong_match_threshold=0.85, 
                                            weak_match_threshold=0.7, 
                                            min_duration_for_new_speaker=2.0):
        """
        Assigns a consistent person name using a tiered confidence system.
        """
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)

        if not self.speaker_voiceprints:
            person_name = f"Person {self.person_counter}"
            self.speaker_voiceprints[person_name] = {'embedding': embedding, 'count': 1}
            self.person_counter += 1
            if self.debug:
                print(f"\n🎤 New speaker identified: {person_name} (first voice)")
            return person_name

        distances = {
            name: cdist(embedding, vp['embedding'], metric='cosine')[0, 0]
            for name, vp in self.speaker_voiceprints.items()
        }

        min_distance = min(distances.values())
        best_match_person = min(distances, key=distances.get)
        similarity = 1 - min_distance

        if similarity >= strong_match_threshold:
            person_name = best_match_person
            self.update_voiceprint(person_name, embedding)
            if self.debug:
                print(f"✅ Strong match: {person_name} (similarity: {similarity:.2f}), voiceprint updated.")
            return person_name
        elif similarity >= weak_match_threshold:
            person_name = best_match_person
            if self.debug:
                print(f"👉 Weak match: {person_name} (similarity: {similarity:.2f}), assigning but not updating voiceprint.")
            return person_name
        elif duration < min_duration_for_new_speaker:
            person_name = best_match_person
            if self.debug:
                print(f"⚠️ Very weak match (sim: {similarity:.2f}), but short duration ({duration:.1f}s). Force-matching to {person_name} without updating.")
            return person_name
        else:
            person_name = f"Person {self.person_counter}"
            self.speaker_voiceprints[person_name] = {'embedding': embedding, 'count': 1}
            self.person_counter += 1
            if self.debug:
                print(f"\n🎤 New speaker identified: {person_name} (similarity to closest match {best_match_person}: {similarity:.2f})")
            return person_name

    def update_voiceprint(self, person_name, embedding):
        """Updates a person's voiceprint with a new embedding using a running average."""
        old_embedding = self.speaker_voiceprints[person_name]['embedding']
        old_count = self.speaker_voiceprints[person_name]['count']
        new_embedding = (old_embedding * old_count + embedding) / (old_count + 1)
        self.speaker_voiceprints[person_name]['embedding'] = new_embedding
        self.speaker_voiceprints[person_name]['count'] += 1

    def _process_realtime_audio(self):
        """Processes audio from the queue for real-time transcription display."""
        while not self.is_stopping:
            try:
                audio_chunk = self.realtime_audio_queue.get(timeout=1)
                
                # Simple VAD: Check if the chunk has enough energy to be considered speech
                rms = np.sqrt(np.mean(audio_chunk**2))
                if rms < self.vad_threshold:
                    continue # Skip silent chunks

                # Run ASR on the small chunk
                result = self.asr_pipeline(audio_chunk)
                text = result["text"].strip()
                
                if text:
                    # De-duplicate with the last transcribed text
                    if self.last_realtime_text:
                        # Find the longest common suffix/prefix
                        for i in range(len(self.last_realtime_text), 0, -1):
                            if text.startswith(self.last_realtime_text[-i:]):
                                text = text[i:]
                                break
                    
                    if text:
                        print(f"[Live]: {text}")
                        self.last_realtime_text = result["text"].strip() # Store the original for next comparison

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in real-time processing: {e}", file=sys.stderr)

    def on_stream_finished(self):
        """Called by the sounddevice thread when the audio stream stops."""
        # In meeting mode the MAIN thread (run_meeting_mode) owns stopping and
        # processing -- crucial because the modern engine uses MLX, whose GPU
        # stream is thread-local and only works on the main thread. So here we
        # just signal; we must NOT kick off processing from this callback thread.
        if self.meeting_mode:
            self.is_recording = False
            return
        if self.is_recording:
            print("Audio stream unexpectedly finished. Cleaning up.")
        self.stop_recording()

    def _close_stream(self):
        """Stop and close the audio input stream (safe to call repeatedly)."""
        self.is_recording = False
        if getattr(self, 'stream', None):
            try:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def audio_callback(self, indata, frames, time, status):
        """
        Callback for audio input.
        - Appends to the main buffer for final processing.
        - Puts small chunks onto a queue for real-time transcription (if enabled).
        """
        if status:
            print(f"Status: {status}", file=sys.stderr)

        if self.is_recording:
            # Downmix to mono. indata is (frames, channels); averaging handles both
            # 1-channel mics and 2-channel loopback/aggregate devices (e.g. BlackHole).
            mono = indata.mean(axis=1).astype(np.float32)

            # Add to the main buffer for high-quality final processing.
            # Store a copy of the chunk (indata is reused by PortAudio) and keep a
            # running sample count instead of growing a sample-level Python list.
            self.audio_buffer.append(mono.copy())
            self.audio_buffer_samples += mono.shape[0]

            # Only process real-time audio if live transcription is enabled
            if self.live_transcription:
                # Add to the real-time buffer for immediate transcription
                self.realtime_buffer.extend(mono)

                # Process real-time buffer if it's full
                if len(self.realtime_buffer) >= self.realtime_buffer_size:
                    chunk = np.array(self.realtime_buffer).astype(np.float32)
                    self.realtime_audio_queue.put(chunk)
                    # Slide the buffer, keeping the overlap for context
                    self.realtime_buffer = self.realtime_buffer[-self.realtime_overlap_size:]

            # Check if max duration is reached for the main recording
            if self.audio_buffer_samples >= self.max_buffer_size:
                print("\nMax recording duration reached. Stopping stream...")
                self.is_recording = False  # Signal the main loop to move on
                raise sd.CallbackStop

    def _ollama_models(self):
        """Preferred summarization models, newest first, with graceful fallbacks.
        Any model not present locally returns 404 and we move to the next one."""
        models = [self.ollama_model, "qwen3:8b", "qwen3:4b", "gemma3:4b", "llama3.2", "gemma2:2b"]
        return list(dict.fromkeys(models))  # de-dupe, preserve order

    def _ollama_available(self):
        """True if the Ollama server is reachable."""
        try:
            requests.get(self.OLLAMA_URL, timeout=5)
            return True
        except requests.exceptions.RequestException:
            return False

    def _ollama_complete(self, prompt, stream=False, print_stream=False, header=None, timeout=180):
        """Run a single prompt through Ollama, trying each preferred model in turn.
        Returns the generated text, or None if every model failed."""
        for model_name in self._ollama_models():
            try:
                response = requests.post(
                    f"{self.OLLAMA_URL}/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": stream},
                    stream=stream,
                    timeout=None if stream else timeout,
                )

                if response.status_code == 404:
                    print(f"Warning: Model '{model_name}' not found. Trying next model.")
                    continue
                response.raise_for_status()

                if stream:
                    if header:
                        print("\n" + "=" * 50 + f"\n{header} (from {model_name})\n" + "=" * 50)
                    text = ""
                    for line in response.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("response", "")
                        if print_stream:
                            print(token, end="", flush=True)
                        text += token
                        if data.get("done") and print_stream:
                            print("\n")
                    text = text.strip()
                else:
                    text = response.json().get("response", "").strip()

                if text:
                    return text

            except requests.exceptions.RequestException as e:
                print(f"Error connecting to Ollama with model {model_name}: {e}")
                continue
            except json.JSONDecodeError:
                print(f"Error decoding response from Ollama with model {model_name}.")
                continue

        return None

    def _chunk_text(self, text, max_chars):
        """Split text on line boundaries into chunks no larger than max_chars."""
        chunks, current, size = [], [], 0
        for line in text.split("\n"):
            if size + len(line) + 1 > max_chars and current:
                chunks.append("\n".join(current))
                current, size = [], 0
            current.append(line)
            size += len(line) + 1
        if current:
            chunks.append("\n".join(current))
        return chunks

    def _reduce_long_transcript(self, transcript):
        """Map step of map-reduce summarization: condense each chunk so the final
        summary prompt fits comfortably in a small model's context window."""
        chunks = self._chunk_text(transcript, self.summary_char_threshold)
        print(f"📚 Long transcript ({len(transcript)} chars); summarizing in {len(chunks)} chunks...")
        partials = []
        for i, chunk in enumerate(chunks, 1):
            print(f"   • Summarizing chunk {i}/{len(chunks)}...")
            part = self._ollama_complete(self.CHUNK_PROMPT.format(transcript=chunk), stream=False)
            if part:
                partials.append(part)
        return "\n\n".join(partials) if partials else transcript

    def generate_summary(self):
        """Generates a summary of the conversation using a local LLM via Ollama."""
        if not self.text_buffer:
            return "No conversation recorded."

        if not self._ollama_available():
            return "Ollama server not running. Please start Ollama to generate a summary."

        print("✨ Generating summary with local LLM via Ollama...")

        full_transcript = "\n".join(self.text_buffer)
        if len(full_transcript) > self.summary_char_threshold:
            full_transcript = self._reduce_long_transcript(full_transcript)

        prompt = self.SUMMARY_PROMPT.format(transcript=full_transcript)
        summary = self._ollama_complete(prompt, stream=True, print_stream=True, header="MEETING SUMMARY")
        return summary or "Failed to generate summary. All models are unavailable."

    def save_meeting_to_file(self, summary):
        """Save the meeting summary and transcription to separate dated files in a transcriptions folder"""
        try:
            # Create transcriptions directory if it doesn't exist
            transcriptions_dir = "transcriptions"
            if not os.path.exists(transcriptions_dir):
                os.makedirs(transcriptions_dir)

            # Generate filename with current date and timestamp
            current_datetime = datetime.now().strftime("%m-%d-%y-%H%M")
            
            # Create separate files for summary and transcription
            summary_filename = f"meeting-summary-{current_datetime}.txt"
            transcription_filename = f"meeting-transcription-{current_datetime}.txt"
            
            # Full paths
            summary_path = os.path.join(transcriptions_dir, summary_filename)
            transcription_path = os.path.join(transcriptions_dir, transcription_filename)

            # If files already exist, add a number suffix
            counter = 1
            base_summary_filename = summary_filename
            base_transcription_filename = transcription_filename
            
            while os.path.exists(summary_path) or os.path.exists(transcription_path):
                name_part_summary = base_summary_filename.replace('.txt', '')
                name_part_transcription = base_transcription_filename.replace('.txt', '')
                summary_filename = f"{name_part_summary}-{counter}.txt"
                transcription_filename = f"{name_part_transcription}-{counter}.txt"
                summary_path = os.path.join(transcriptions_dir, summary_filename)
                transcription_path = os.path.join(transcriptions_dir, transcription_filename)
                counter += 1

            # Save summary file
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("MEETING SUMMARY\n")
                f.write(f"Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
                f.write(f"ASR Model: {self.asr_label}\n")
                f.write(f"Diarization: {self.diar_label}\n")
                f.write("="*60 + "\n\n")
                f.write(summary + "\n")
                f.write("\n" + "="*60 + "\n")
                f.write("End of Meeting Summary\n")
                f.write("="*60 + "\n")

            # Save transcription file
            with open(transcription_path, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("MEETING TRANSCRIPTION\n")
                f.write(f"Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
                f.write(f"ASR Model: {self.asr_label}\n")
                f.write(f"Diarization: {self.diar_label}\n")
                f.write("="*60 + "\n\n")
                if self.text_buffer:
                    for line in self.text_buffer:
                        f.write(line + "\n")
                else:
                    f.write("No transcription recorded.\n")
                f.write("\n" + "="*60 + "\n")
                f.write("End of Meeting Transcription\n")
                f.write("="*60 + "\n")

            # Return both filenames for display
            return f"{summary_filename} and {transcription_filename}"

        except Exception as e:
            print(f"Error saving meeting to file: {str(e)}", file=sys.stderr)
            return None

    def start_recording(self):
        """Start the recording process and the real-time transcription thread."""
        # Ensure any previous recording is fully stopped
        if hasattr(self, 'stream') and self.stream is not None:
            print("Waiting for previous recording to fully stop...")
            time.sleep(2)  # Give time for previous recording to fully stop
        
        self.is_recording = True
        self.is_stopping = False  # Reset stopping flag
        self.audio_buffer = []  # Clear the audio buffer for new recording
        self.audio_buffer_samples = 0
        self.last_realtime_text = ""  # Reset the real-time text buffer

        # Start the real-time processing thread only if live transcription is enabled
        if self.live_transcription:
            self.realtime_thread = threading.Thread(target=self._process_realtime_audio)
            self.realtime_thread.start()

        # Resolve the input device and its channel count (we downmix to mono).
        try:
            dev_info = sd.query_devices(self.input_device, 'input')
            # Capture ALL input channels (capped for safety) and downmix to mono in
            # the callback. This matters for Aggregate Devices (e.g. mic + BlackHole),
            # where the mic may be channel 0 or the last channel -- capping at 2 could
            # silently drop it.
            self.channels = min(8, int(dev_info['max_input_channels'])) or 1
            print(f"🎚️  Input device: {dev_info['name']} ({self.channels}ch)")
            # Loopback/aggregate devices (BlackHole, Zoom) often run at 48kHz, not
            # 16kHz. If 16kHz isn't supported, capture at the device default and let
            # the modern engine resample to 16kHz.
            try:
                sd.check_input_settings(device=self.input_device, channels=self.channels,
                                        samplerate=self.sample_rate, dtype=self.dtype)
            except Exception:
                self.sample_rate = int(dev_info['default_samplerate'])
                self.max_buffer_size = int(self.sample_rate * self.max_duration_seconds)
                print(f"ℹ️  16kHz unsupported here; capturing at {self.sample_rate}Hz (auto-resampled).")
        except Exception as e:
            print(f"⚠️  Could not query input device ({e}); using default mono input.")
            self.channels = 1

        # Start the main audio stream
        print("Starting audio stream...")
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=self.input_device,
                callback=self.audio_callback,
                finished_callback=self.on_stream_finished
            )
            self.stream.start()
            print(f"\n🎙️  Recording started. Press Ctrl+C or wait for {self.max_duration_seconds / 60:.0f} minutes to stop.")
            if self.live_transcription:
                print("📺 Live transcription preview is enabled.")
            else:
                print("📺 Live transcription preview is disabled. Only final transcript will be generated.")
            print()
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.is_recording = False
            raise

    def stop_recording(self, wait_for_processing=False, processing_timeout=30):
        """Stop the recording and trigger processing of the entire buffer.

        processing_timeout: seconds to block when wait_for_processing is True;
        pass None to wait until processing fully completes (used in meeting mode,
        where a long recording can take minutes to transcribe and summarize)."""
        if self.is_stopping:
            return
        self.is_stopping = True

        # This message is more of a "processing started" message now
        print("\nRecording stopped. Initiating background processing...")
        self.is_recording = False

        # Stop the real-time thread only if it exists (live transcription was enabled)
        if self.live_transcription and hasattr(self, 'realtime_thread'):
            self.realtime_thread.join(timeout=2.0)

        # Stop the audio stream - make it safe to call on an already stopped stream
        if hasattr(self, 'stream') and self.stream:
            if self.stream.active:
                self.stream.stop()
            self.stream.close()
            self.stream = None  # Clear the stream reference

        # Start background processing in a separate daemon thread
        self.processing_thread = threading.Thread(target=self._background_processing, daemon=True)
        self.processing_thread.start()

        # If requested, wait for processing to complete
        if wait_for_processing:
            self.processing_thread.join(timeout=processing_timeout)
            print("Background processing completed.")
        elif self.meeting_mode:
            print("\n⏳ Processing the full meeting recording (transcription + summary)...")
        else:
            print("\nRecording stopped. Processing will continue in the background.")
            print("You can start a new recording or press Ctrl+C to exit.")

    def wait_for_processing(self):
        """Block until any in-flight background processing finishes (no timeout)."""
        thread = getattr(self, 'processing_thread', None)
        if thread is not None:
            thread.join()

    def _background_processing(self):
        """Process the audio buffer in the background."""
        try:
            print("\n🔄 Processing recorded audio...")
            # Process the entire audio buffer at once
            self.process_audio_buffer()

            # Generate and display the final summary
            print("\n📝 Generating summary...")
            summary = self.generate_summary()

            # Save to file
            filename = self.save_meeting_to_file(summary)
            if filename:
                print("\n" + "="*68)
                print(f"✅ Meeting files saved to transcriptions/ folder:")
                print(f"   📄 Summary: {filename.split(' and ')[0]}")
                print(f"   📝 Transcription: {filename.split(' and ')[1]}")
                print("="*68 + "\n")

        except Exception as e:
            print(f"Error during summary generation: {str(e)}")
            # Still try to save what we have
            try:
                print("\n💾 Saving transcription without summary...")
                filename = self.save_meeting_to_file("Summary generation failed.")
                if filename:
                    print(f"✅ Transcription saved to: {filename}")
            except Exception as save_error:
                print(f"❌ Failed to save transcription: {str(save_error)}")

    def create_summary_of_summaries(self):
        """Create a comprehensive summary of all meeting summaries from the current session using Ollama."""
        try:
            transcriptions_dir = "transcriptions"
            if not os.path.exists(transcriptions_dir):
                print("No transcriptions folder found. Skipping summary of summaries.")
                return

            # Find all summary files created during this session
            summary_files = []
            for filename in os.listdir(transcriptions_dir):
                if filename.startswith("meeting-summary-") and filename.endswith(".txt"):
                    file_path = os.path.join(transcriptions_dir, filename)
                    # Get file creation time for chronological ordering
                    creation_time = os.path.getctime(file_path)
                    creation_datetime = datetime.fromtimestamp(creation_time)
                    
                    # Only include files created during this session
                    if creation_datetime >= self.session_start_time:
                        summary_files.append((creation_time, file_path, filename))

            if not summary_files:
                print("No summary files found from current session. Skipping summary of summaries.")
                return

            # Sort by creation time (chronological order)
            summary_files.sort(key=lambda x: x[0])

            # Read all summaries
            all_summaries = []
            combined_summaries_text = ""
            
            for creation_time, file_path, filename in summary_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Extract just the summary content (between the header and footer)
                        lines = content.split('\n')
                        summary_start = False
                        summary_content = []
                        
                        for line in lines:
                            if "MEETING SUMMARY" in line:
                                summary_start = True
                                continue
                            elif "End of Meeting Summary" in line:
                                break
                            elif summary_start and line.strip():
                                summary_content.append(line)
                        
                        if summary_content:
                            # Get the date from the filename or content
                            meeting_date = filename.replace("meeting-summary-", "").replace(".txt", "")
                            summary_text = '\n'.join(summary_content).strip()
                            
                            all_summaries.append({
                                'date': meeting_date,
                                'filename': filename,
                                'content': summary_text
                            })
                            
                            # Add to combined text for Ollama processing
                            combined_summaries_text += f"\n\nMEETING {len(all_summaries)} - {meeting_date}:\n{summary_text}"
                            
                except Exception as e:
                    print(f"Error reading summary file {filename}: {e}")

            if not all_summaries:
                print("No valid summaries found. Skipping summary of summaries.")
                return

            # Use Ollama to create an intelligent summary of summaries
            print("🤖 Generating intelligent session summary with Ollama...")
            session_summary = self.generate_session_summary(combined_summaries_text, len(all_summaries))

            # Create the comprehensive summary
            current_datetime = datetime.now().strftime("%m-%d-%y-%H%M")
            full_summary_filename = f"full-meeting-summary-{current_datetime}.txt"
            full_summary_path = os.path.join(transcriptions_dir, full_summary_filename)

            # Check if file already exists and add suffix if needed
            counter = 1
            base_filename = full_summary_filename
            while os.path.exists(full_summary_path):
                name_part = base_filename.replace('.txt', '')
                full_summary_filename = f"{name_part}-{counter}.txt"
                full_summary_path = os.path.join(transcriptions_dir, full_summary_filename)
                counter += 1

            with open(full_summary_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("SESSION SUMMARY OF SUMMARIES\n")
                f.write(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
                f.write(f"Session Start: {self.session_start_time.strftime('%B %d, %Y at %I:%M %p')}\n")
                f.write(f"Total Meetings: {len(all_summaries)}\n")
                f.write("="*80 + "\n\n")

                # Write the AI-generated session summary
                f.write("AI-GENERATED SESSION OVERVIEW\n")
                f.write("-" * 50 + "\n")
                f.write(session_summary)
                f.write("\n\n")

                # Write individual meeting summaries
                f.write("INDIVIDUAL MEETING SUMMARIES\n")
                f.write("-" * 50 + "\n")
                for i, summary in enumerate(all_summaries, 1):
                    f.write(f"MEETING {i} - {summary['date']}\n")
                    f.write("-" * 30 + "\n")
                    f.write(summary['content'])
                    f.write("\n\n")

                f.write("="*80 + "\n")
                f.write("End of Session Summary\n")
                f.write("="*80 + "\n")

            print(f"\n📋 Session Summary created: {full_summary_filename}")
            print(f"   📁 Location: {transcriptions_dir}/")
            print(f"   📊 Contains summaries from {len(all_summaries)} meetings")
            print(f"   🤖 AI-generated session overview included")

        except Exception as e:
            print(f"Error creating summary of summaries: {e}")

    def generate_session_summary(self, combined_summaries_text, num_meetings):
        """Generate an intelligent summary of the session using Ollama."""
        try:
            # Check if Ollama server is running
            try:
                requests.get("http://localhost:11434")
            except requests.exceptions.ConnectionError:
                return "Ollama server not running. Please start Ollama to generate an intelligent session summary."

            prompt = f"""
You are an expert meeting analyst. Your task is to create a comprehensive overview of a session containing {num_meetings} meetings.

Please analyze the following meeting summaries and provide:
1. A high-level overview of the entire session
2. Key themes and topics that emerged across all meetings
3. Important decisions or action items identified
4. Any patterns or trends you notice
5. A brief conclusion about the session's overall effectiveness

Here are the meeting summaries:
{combined_summaries_text}

Please provide a well-structured, professional summary that captures the essence of this entire session.
"""

            # Define the models to try in order of preference
            models_to_try = [self.ollama_model, "llama3.2", "gemma2:2b"]
            # Remove duplicates, keeping the order
            models_to_try = list(dict.fromkeys(models_to_try))

            for model_name in models_to_try:
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": model_name,
                            "prompt": prompt,
                            "stream": False
                        },
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        summary = result.get("response", "").strip()
                        if summary:
                            return summary
                    else:
                        print(f"Error from Ollama with model {model_name}: {response.status_code}")
                        continue

                except requests.exceptions.RequestException as e:
                    print(f"Error connecting to Ollama with model {model_name}: {e}")
                    continue
                except json.JSONDecodeError:
                    print(f"Error decoding response from Ollama with model {model_name}.")
                    continue

            return "Failed to generate session summary. All models are unavailable."

        except Exception as e:
            return f"Error generating session summary: {str(e)}"

def run_meeting_mode(transcriber):
    """Passive meeting capture: record one continuous session until Ctrl+C (or the
    safety cap), then process the whole recording once into a single transcript and
    summary. No 5-minute chunking, no per-chunk files, no audio gaps."""
    print("=" * 68)
    print("🎧 PASSIVE MEETING CAPTURE")
    print("   Recording the whole meeting. Press Ctrl+C when it ends to stop and")
    print("   generate your transcript + summary.")
    print(f"   (Safety auto-stop after {transcriber.max_duration_seconds / 60:.0f} minutes.)")
    print("=" * 68 + "\n")

    try:
        transcriber.start_recording()
        # Wait until recording stops, either via Ctrl+C or the safety cap.
        while transcriber.is_recording:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    print("\n🛑 Meeting ended. Stopping recording and processing the full session...")
    # Stop the stream, then process on THIS (main) thread. The modern engine uses
    # MLX, which can only reach the GPU from the main thread -- running processing
    # in a background thread is what caused the "no Stream(gpu, 0)" failure.
    transcriber._close_stream()
    try:
        transcriber._background_processing()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted while processing; results may be incomplete.")

    print("\n✅ Done. Your transcript and summary are saved in the transcriptions/ folder.")
    print("Goodbye!")


def main():
    """Main function to run the meeting transcriber"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Meeting Transcriber with ASR + Diarization')
    parser.add_argument('--live', '--live-transcription', action='store_true',
                       help='Enable live transcription preview (disabled by default)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for detailed processing output')
    parser.add_argument('--model', default='openai/whisper-medium.en',
                       help='Whisper model to use (default: openai/whisper-medium.en)')
    parser.add_argument('--ollama-model', default='qwen3:8b',
                       help='Ollama model for summarization (default: qwen3:8b)')
    parser.add_argument('--modern', action='store_true',
                       help='Use the modern engine: Parakeet-MLX ASR + pyannote community-1 '
                            '(requires the venv-modern environment)')
    parser.add_argument('--meeting', action='store_true',
                       help='Passive meeting capture: one continuous recording until Ctrl+C, '
                            'then a single transcript + summary (uses the modern engine)')
    parser.add_argument('--max-minutes', type=float, default=None,
                       help='Safety auto-stop duration (default: 180 in meeting mode, 5 otherwise)')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio input devices and exit')
    parser.add_argument('--device', default=None,
                       help='Input device to record from: an index (e.g. 3) or a name '
                            'substring (e.g. BlackHole). Default: system default input.')
    parser.add_argument('--from-audio', default=None,
                       help='Re-transcribe a saved WAV file instead of recording from the mic')
    parser.add_argument('--from-transcript', default=None,
                       help='Re-summarize an existing transcript .txt with --ollama-model (no ASR)')

    args = parser.parse_args()

    # List devices and exit (handy for finding your loopback/aggregate device).
    if args.list_devices:
        import sounddevice as sd
        print("Available input devices:")
        default_in = sd.default.device[0]
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                mark = "  <-- default" if i == default_in else ""
                print(f"  [{i}] {d['name']} ({d['max_input_channels']}ch){mark}")
        return

    # Resolve --device (index or name substring) to a device index.
    input_device = None
    if args.device is not None:
        import sounddevice as sd
        if args.device.isdigit():
            input_device = int(args.device)
        else:
            matches = [i for i, d in enumerate(sd.query_devices())
                       if d['max_input_channels'] > 0 and args.device.lower() in d['name'].lower()]
            if not matches:
                print(f"Error: no input device matching '{args.device}'. Use --list-devices.")
                sys.exit(1)
            input_device = matches[0]
        print(f"🎚️  Selected input device index: {input_device}")

    # Meeting mode and re-transcription use the modern engine by default.
    use_modern = args.modern or args.meeting or bool(args.from_audio)
    summary_only = bool(args.from_transcript)

    # Whisper Model Options:
    # - "openai/whisper-large-v3": Best accuracy, latest model
    # - "openai/whisper-large-v2": Previous best model, excellent accuracy
    # - "openai/whisper-medium.en": Good for English-only meetings
    # - "openai/whisper-base": Faster but lower accuracy
    
    # Set debug=True to see detailed processing output
    transcriber = MeetingTranscriber(
        model_name=args.model,
        debug=args.debug,
        ollama_model=args.ollama_model,
        live_transcription=args.live,
        use_modern=use_modern,
        meeting_mode=args.meeting,
        max_minutes=args.max_minutes,
        input_device=input_device,
        summary_only=summary_only
    )

    # Re-summarize an existing transcript with a different Ollama model, then exit.
    if args.from_transcript:
        if not os.path.exists(args.from_transcript):
            print(f"Error: transcript file not found: {args.from_transcript}")
            sys.exit(1)
        transcriber.process_transcript_file(args.from_transcript)
        return

    print(f"🎙️  Meeting Transcriber - ASR + Diarization Pipeline")
    print(f"🤖  Using ASR model: {transcriber.asr_label}")
    print(f"🎯  Using Diarization: {transcriber.diar_label}")
    print(f"⏱️  Recording will stop automatically after {transcriber.max_duration_seconds / 60:.0f} minutes.")
    if args.live:
        print(f"📺 Live transcription preview: ENABLED")
    else:
        print(f"📺 Live transcription preview: DISABLED (use --live to enable)")
    print()

    # Re-transcribe a saved recording (no microphone) and exit.
    if args.from_audio:
        if not os.path.exists(args.from_audio):
            print(f"Error: audio file not found: {args.from_audio}")
            sys.exit(1)
        transcriber.process_audio_file(args.from_audio)
        print("\n✅ Done. Transcript and summary saved in transcriptions/.")
        return

    # Passive meeting capture: one continuous recording, processed once on stop.
    if transcriber.meeting_mode:
        run_meeting_mode(transcriber)
        return

    try:
        while True:
            try:
                transcriber.start_recording()
                # Wait until recording is stopped (either by duration or Ctrl+C)
                while transcriber.is_recording:
                    time.sleep(0.2)
                
                # If we are here, it means is_recording is False.
                # stop_recording would have been called by on_stream_finished or KeyboardInterrupt
                print("\nReady for next recording in 2 seconds... (Press Ctrl+C to exit)")
                time.sleep(2)

            except KeyboardInterrupt:
                print("\nCtrl+C received, processing current recording...")
                # If we're currently recording, stop and process it
                if transcriber.is_recording:
                    transcriber.stop_recording(wait_for_processing=True)
                break # Exit the main loop

            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                transcriber.stop_recording(wait_for_processing=True) # Try to cleanup
                time.sleep(2)  # Wait before retrying
                continue

    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        print("\nApplication shutting down.")
        # Ensure any remaining recording is processed
        if transcriber.is_recording:
            print("Processing final recording...")
            transcriber.stop_recording(wait_for_processing=True)
        
        # Create summary of summaries at the end of the session
        print("\n📋 Creating session summary...")
        transcriber.create_summary_of_summaries()
        print("Goodbye!")


if __name__ == "__main__":
    main()