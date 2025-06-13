#!/usr/bin/env python3
"""
Meeting Transcriber - Using Hugging Face ASR + Diarization Pipeline
Based on: https://huggingface.co/blog/asr-diarization

Uses an integrated ASR + Diarization pipeline for accurate speaker identification.
"""

import os
import sys
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

class MeetingTranscriber:
    def __init__(self, model_name="openai/whisper-large-v3", debug=False):
        """
        Initialize the meeting transcriber.

        Args:
            model_name (str): Whisper model to use
            debug (bool): Whether to enable debug mode
        """
        self.debug = debug
        self.whisper_model_name = model_name

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
        self.max_duration_seconds = 10 * 60  # 10 minutes
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
            
            # Initialize summarization pipeline
            self.summarizer = pipeline("summarization", model="knkarthick/meeting-summary-samsum")

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
                    for segment in segments:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"\n[{timestamp}] 👤 {segment['speaker']}: {segment['text']}")
                        self.text_buffer.append(f"{segment['speaker']}: {segment['text']}")
            finally:
                os.unlink(self.temp_audio_file)

        except Exception as e:
            print(f"Error processing audio: {str(e)}", file=sys.stderr)

    def get_embedding(self, audio_path, segment):
        """Extracts an embedding for a given audio file path and speaker segment."""
        try:
            inference = Inference(self.embedding_model, window="whole")
            # The 'crop' method takes the file path and the segment to extract
            embedding = inference.crop(audio_path, segment)
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}", file=sys.stderr)
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
                print("\nMax recording duration reached. Stopping recording...")
                self.stop_recording()

    def generate_summary(self):
        """Generate a summary of the conversation."""
        if not self.text_buffer:
            return "No conversation recorded."

        # Get unique speakers and their messages
        speaker_messages = {}
        for line in self.text_buffer:
            try:
                speaker, message = line.split(':', 1)
                speaker = speaker.strip()
                if speaker not in speaker_messages:
                    speaker_messages[speaker] = []
                speaker_messages[speaker].append(message.strip())
            except ValueError:
                continue

        if not speaker_messages:
            return "No speakers identified in the conversation."

        # Generate the summary
        summary_parts = []

        # High-level summary
        speakers = list(speaker_messages.keys())
        if len(speakers) == 1:
            summary_parts.append(f"{speakers[0]} was the only speaker in the conversation.")
        else:
            main_speaker = speakers[0]
            other_speakers = [s for s in speakers[1:] if s != main_speaker]
            if other_speakers:
                summary_parts.append(f"{main_speaker} had a conversation with {', '.join(other_speakers)}.")
            else:
                summary_parts.append(f"{main_speaker} was the main speaker in the conversation.")

        # Individual speaker summaries
        summary_parts.append("\nSpeaker Summaries:")
        for speaker, messages in speaker_messages.items():
            # Combine all messages for this speaker
            combined_text = " ".join(messages)

            # Generate a summary of this speaker's contributions
            try:
                if len(combined_text) > 100:
                    # Calculate appropriate max_length based on input length
                    input_words = len(combined_text.split())
                    max_length = max(20, min(80, int(input_words * 0.7)))
                    min_length = max(10, min(max_length - 5, 15))

                    summary = self.summarizer(combined_text,
                                           max_length=max_length,
                                           min_length=min_length,
                                           do_sample=False)
                    speaker_summary = summary[0]['summary_text']
                else:
                    speaker_summary = combined_text

                summary_parts.append(f"\n{speaker}:")
                summary_parts.append(f"  {speaker_summary}")
            except Exception as e:
                print(f"Error summarizing {speaker}'s contributions: {str(e)}", file=sys.stderr)
                summary_parts.append(f"\n{speaker}:")
                summary_parts.append(f"  {combined_text}")

        return "\n".join(summary_parts)

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
        self.is_recording = True

        # Start the real-time processing thread
        self.realtime_thread = threading.Thread(target=self._process_realtime_audio)
        self.realtime_thread.start()

        # Start the main audio stream
        print("Starting audio stream...")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self.audio_callback
        )
        self.stream.start()
        print(f"\n🎙️  Recording started. Press Ctrl+C or wait for {self.max_duration_seconds / 60:.0f} minutes to stop.\n")

    def stop_recording(self):
        """Stop the recording and trigger processing of the entire buffer."""
        if self.is_stopping:
            return
        self.is_stopping = True

        print("\nStopping recording...")
        self.is_recording = False

        # Stop the real-time thread
        if hasattr(self, 'realtime_thread'):
            self.realtime_thread.join(timeout=2.0)

        # Stop the audio stream
        if hasattr(self, 'stream') and self.stream.active:
            self.stream.stop()
            self.stream.close()
        
        # Process the entire audio buffer at once
        self.process_audio_buffer()

        try:
            # Generate and display the final summary
            print("\n📝 Generating final summary...")
            summary = self.generate_summary()
            print("\n" + "="*50)
            print("MEETING SUMMARY")
            print("="*50)
            print(summary)
            print("="*50)

            # Save to file
            print("\n💾 Saving meeting to file...")
            filename = self.save_meeting_to_file(summary)
            if filename:
                print(f"✅ Meeting saved to: {filename}")
            else:
                print("❌ Failed to save meeting to file")

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

        print("\nRecording stopped. Goodbye!")


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
        debug=True  # Enable to see detailed processing
    )
    
    print(f"🎙️  Meeting Transcriber - ASR + Diarization Pipeline")
    print(f"🤖  Using Whisper model: {transcriber.whisper_model_name}")
    print(f"🎯  Using Diarization: pyannote/speaker-diarization-3.1")
    print(f"⏱️  Recording will stop automatically after {transcriber.max_duration_seconds / 60:.0f} minutes.")
    
    try:
        transcriber.start_recording()
        while transcriber.is_recording:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C received, shutting down...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        try:
                transcriber.stop_recording()
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
            # Force exit if shutdown fails
            import sys
            sys.exit(1)


if __name__ == "__main__":
    main()