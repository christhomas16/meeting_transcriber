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

# Suppress all informational and warning messages from PyTorch Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Suppress specific warnings from other libraries that are less aggressive
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.pipelines.speaker_verification")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.tasks.segmentation.mixins")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.model")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", message=".*`ModelCheckpoint` callback states.*")

import time
import queue
import threading
import numpy as np
import sounddevice as sd
import torch
from datetime import datetime
from dotenv import load_dotenv
from transformers import pipeline
from pyannote.audio import Pipeline
from pyannote.audio import Model as PyannoteModel
from pyannote.audio import Inference
from scipy.spatial.distance import cdist
import tempfile
import soundfile as sf
import requests
import json

class MeetingTranscriber:
    def __init__(self, model_name="openai/whisper-medium.en", debug=False, ollama_model="llama3.2"):
        """
        Initialize the meeting transcriber.

        Args:
            model_name (str): Whisper model to use
            debug (bool): Whether to enable debug mode
            ollama_model (str): The name of the Ollama model to use for summarization
        """
        self.debug = debug
        self.whisper_model_name = model_name
        self.ollama_model = ollama_model

        # Load environment variables and check for token
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("\nError: Hugging Face token not found. Please see README.md for setup instructions.")
            sys.exit(1)

        # Audio parameters
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        self.is_recording = False
        self.is_stopping = False

        # Main buffer for final, high-quality processing
        self.audio_buffer = []
        self.max_duration_seconds = 5 * 60  # 5 minutes
        self.max_buffer_size = int(self.sample_rate * self.max_duration_seconds)

        # Real-time processing components
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
        
        # Speaker identification components
        self.person_counter = 1
        self.speaker_voiceprints = {}

        # --- Model Initialization ---
        print("Initializing ASR + Diarization pipeline...")
        try:
            # Initialize ASR pipeline
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
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
            
            self.device_info = device
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

        print(f"\nProcessing {len(self.audio_buffer) / self.sample_rate:.1f}s of recorded audio...")
        audio_data = np.array(self.audio_buffer).astype(np.float32)

        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data /= np.max(np.abs(audio_data))
        
        try:
            # Save audio to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, self.sample_rate)
                self.temp_audio_file = temp_file.name # Store path for embedding model

            try:
                # Run diarization
                if self.debug: print("🎯 Running diarization...")
                diarization = self.diarization_pipeline(self.temp_audio_file)

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
            finally:
                os.unlink(self.temp_audio_file)

        except Exception as e:
            print(f"Error processing audio: {str(e)}", file=sys.stderr)

    def get_embedding(self, audio_path, segment):
        """Extracts an embedding for a given audio file path and speaker segment."""
        # Segments shorter than a certain duration can cause errors in the embedding model.
        MIN_DURATION = 0.05  # 50 milliseconds, a safe threshold for pyannote/embedding
        if segment.duration < MIN_DURATION:
            if self.debug:
                print(f"⏩ Skipping embedding for very short segment ({segment.duration:.3f}s)", file=sys.stderr)
            return None
            
        try:
            inference = Inference(self.embedding_model, window="whole")
            # The 'crop' method takes the file path and the segment to extract
            embedding = inference.crop(audio_path, segment)
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

        # Aggregate embeddings for each speaker label in the current chunk
        chunk_embeddings = {}
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if speaker_label not in chunk_embeddings:
                chunk_embeddings[speaker_label] = []
            
            embedding = self.get_embedding(self.temp_audio_file, turn)
            
            if embedding is not None:
                chunk_embeddings[speaker_label].append(embedding)

        # Create a mapping from chunk speaker labels to consistent person names
        chunk_speaker_map = {}
        for speaker_label, embeddings in chunk_embeddings.items():
            if not embeddings:
                continue
            avg_embedding = np.mean(embeddings, axis=0)
            duration = sum(turn.end - turn.start for turn, _, lbl in diarization.itertracks(yield_label=True) if lbl == speaker_label)
            person_name = self.get_consistent_person_name_by_voice(avg_embedding, duration)
            chunk_speaker_map[speaker_label] = person_name

        # Create a speaker timeline using the consistent names
        speaker_timeline = []
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if speaker_label in chunk_speaker_map:
                person_name = chunk_speaker_map[speaker_label]
                speaker_timeline.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': person_name
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
        """Called when the audio stream is stopped (e.g., by CallbackStop)."""
        # This is called by the sounddevice thread when the stream is stopped.
        # We can now safely call our stop_recording logic.
        if self.is_recording:
            print("Audio stream unexpectedly finished. Cleaning up.")
        self.stop_recording()

    def audio_callback(self, indata, frames, time, status):
        """
        Callback for audio input.
        - Appends to the main buffer for final processing.
        - Puts small chunks onto a queue for real-time transcription.
        """
        if status:
            print(f"Status: {status}", file=sys.stderr)

        if self.is_recording:
            # Add to the main buffer for high-quality final processing
            self.audio_buffer.extend(indata.flatten())
            
            # Add to the real-time buffer for immediate transcription
            self.realtime_buffer.extend(indata.flatten())

            # Process real-time buffer if it's full
            if len(self.realtime_buffer) >= self.realtime_buffer_size:
                chunk = np.array(self.realtime_buffer).astype(np.float32)
                self.realtime_audio_queue.put(chunk)
                # Slide the buffer, keeping the overlap for context
                self.realtime_buffer = self.realtime_buffer[-self.realtime_overlap_size:]

            # Check if max duration is reached for the main recording
            if len(self.audio_buffer) >= self.max_buffer_size:
                print("\nMax recording duration reached. Stopping stream...")
                self.is_recording = False  # Signal the main loop to move on
                raise sd.CallbackStop

    def generate_summary(self):
        """Generates a summary of the conversation using a local LLM via Ollama."""
        if not self.text_buffer:
            return "No conversation recorded."

        full_transcript = "\n".join(self.text_buffer)

        # Check if Ollama server is running
        try:
            requests.get("http://localhost:11434")
        except requests.exceptions.ConnectionError:
            return "Ollama server not running. Please start Ollama to generate a summary."

        print("✨ Generating summary with local LLM via Ollama...")

        prompt = f"""
You are an expert meeting summarizer. Your task is to provide a concise, easy-to-read summary of the following meeting transcript.

Please identify the main topics of discussion and any key decisions or action items. Structure the summary with a brief overview, followed by bullet points for the main topics.

Here is the transcript:
---
{full_transcript}
---
"""
        # Define the models to try in order of preference
        models_to_try = [self.ollama_model, "llama3.2", "gemma2:2b"]
        # Remove duplicates, keeping the order
        models_to_try = list(dict.fromkeys(models_to_try))

        summary = ""
        
        for model_name in models_to_try:
            print(f"Attempting summarization with model: {model_name}...")
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": True
            }

            try:
                response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
                
                # Check for model not found error specifically
                if response.status_code == 404:
                    try:
                        error_data = response.json()
                        if "model" in error_data.get("error", ""):
                            print(f"Warning: Model '{model_name}' not found. Trying next model.")
                            continue # Try the next model in the list
                    except json.JSONDecodeError:
                        # If response is not JSON, it's a different 404 error
                        pass
                
                # Raise any other HTTP errors
                response.raise_for_status()
                
                # If successful, process the stream
                print("\n" + "="*50)
                print(f"MEETING SUMMARY (from {model_name})")
                print("="*50)
                
                current_summary = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = json.loads(line.decode('utf-8'))
                        token = decoded_line.get("response", "")
                        print(token, end="", flush=True)
                        current_summary += token
                        if decoded_line.get("done"):
                            print("\n") # Add a final newline
                
                summary = current_summary.strip()
                break # Exit the loop on success

            except requests.exceptions.RequestException as e:
                print(f"Error connecting to Ollama with model {model_name}: {e}")
                continue # Try the next model
            except json.JSONDecodeError:
                print(f"Error decoding response from Ollama with model {model_name}.")
                continue # Try the next model
        
        if not summary:
            return "Failed to generate summary. Both primary and fallback models are unavailable."
            
        return summary

    def save_meeting_to_file(self, summary):
        """Save the meeting summary and transcription to a dated file"""
        try:
            # Generate filename with current date and timestamp
            current_datetime = datetime.now().strftime("%m-%d-%y-%H%M")
            filename = f"meeting-transcription-{current_datetime}.txt"

            # If file already exists, add a number suffix
            counter = 1
            base_filename = filename
            while os.path.exists(filename):
                name_part = base_filename.replace('.txt', '')
                filename = f"{name_part}-{counter}.txt"
                counter += 1

            self.output_filename = filename

            with open(filename, 'w', encoding='utf-8') as f:
                # Write header
                f.write("="*60 + "\n")
                f.write("MEETING TRANSCRIPTION\n")
                f.write(f"Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
                f.write(f"ASR Model: {self.whisper_model_name}\n")
                f.write("Diarization: pyannote/speaker-diarization-3.1\n")
                f.write("="*60 + "\n\n")

                # Write summary at the top
                f.write("MEETING SUMMARY\n")
                f.write("-" * 30 + "\n")
                f.write(summary + "\n\n")

                # Write full transcription at the bottom
                f.write("FULL TRANSCRIPTION\n")
                f.write("-" * 30 + "\n")
                if self.text_buffer:
                    for line in self.text_buffer:
                        f.write(line + "\n")
                else:
                    f.write("No transcription recorded.\n")

                f.write("\n" + "="*60 + "\n")
                f.write("End of Meeting Transcription\n")
                f.write("="*60 + "\n")

            return filename

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
        self.last_realtime_text = ""  # Reset the real-time text buffer

        # Start the real-time processing thread
        self.realtime_thread = threading.Thread(target=self._process_realtime_audio)
        self.realtime_thread.start()

        # Start the main audio stream
        print("Starting audio stream...")
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self.audio_callback,
                finished_callback=self.on_stream_finished
            )
            self.stream.start()
            print(f"\n🎙️  Recording started. Press Ctrl+C or wait for {self.max_duration_seconds / 60:.0f} minutes to stop.\n")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.is_recording = False
            raise

    def stop_recording(self):
        """Stop the recording and trigger processing of the entire buffer."""
        if self.is_stopping:
            return
        self.is_stopping = True

        # This message is more of a "processing started" message now
        print("\nRecording stopped. Initiating background processing...")
        self.is_recording = False

        # Stop the real-time thread
        if hasattr(self, 'realtime_thread'):
            self.realtime_thread.join(timeout=2.0)

        # Stop the audio stream - make it safe to call on an already stopped stream
        if hasattr(self, 'stream') and self.stream:
            if self.stream.active:
                self.stream.stop()
            self.stream.close()
            self.stream = None  # Clear the stream reference
        
        # Start background processing in a separate daemon thread
        processing_thread = threading.Thread(target=self._background_processing, daemon=True)
        processing_thread.start()
        
        print("\nRecording stopped. Processing will continue in the background.")
        print("You can start a new recording or press Ctrl+C to exit.")

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
                print(f"✅ Meeting successfully saved to: {filename}")
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


def main():
    """Main function to run the meeting transcriber"""
    
    # Whisper Model Options:
    # - "openai/whisper-large-v3": Best accuracy, latest model
    # - "openai/whisper-large-v2": Previous best model, excellent accuracy
    # - "openai/whisper-medium.en": Good for English-only meetings
    # - "openai/whisper-base": Faster but lower accuracy
    
    # Set debug=True to see detailed processing output
    transcriber = MeetingTranscriber(
        model_name="openai/whisper-medium.en",  # English-only for better accuracy and speed
        debug=False,  # Set to True to see detailed processing
        ollama_model="llama3.2" # Default local model. Change to "gemma2:2b" or a more powerful one like "magistral" if needed.
    )
    
    print(f"🎙️  Meeting Transcriber - ASR + Diarization Pipeline")
    print(f"🤖  Using Whisper model: {transcriber.whisper_model_name}")
    print(f"🎯  Using Diarization: pyannote/speaker-diarization-3.1")
    print(f"⏱️  Recording will stop automatically after {transcriber.max_duration_seconds / 60:.0f} minutes.")
    
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
                print("\nCtrl+C received, shutting down...")
                if transcriber.is_recording:
                    transcriber.stop_recording()
                break # Exit the main loop

            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                transcriber.stop_recording() # Try to cleanup
                time.sleep(2)  # Wait before retrying
                continue

    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        print("\nApplication shutting down.")
        if transcriber.is_stopping is False:
             transcriber.stop_recording()


if __name__ == "__main__":
    main()