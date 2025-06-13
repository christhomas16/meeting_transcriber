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
import contextlib
import numpy as np
import sounddevice as sd
import torch
from datetime import datetime
from dotenv import load_dotenv
from transformers import pipeline
from pyannote.audio import Pipeline
import base64
import tempfile
import soundfile as sf

class MeetingTranscriber:
    def __init__(self, model_name="openai/whisper-large-v3", debug=False):
        """
        Initialize the meeting transcriber with integrated ASR + Diarization.

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
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.is_stopping = False

        # Audio buffer - hybrid approach: start short, then go long
        self.audio_buffer = []
        self.initial_chunk_duration = 30  # Start with 30-second chunks for immediate feedback
        self.full_chunk_duration = 15 * 60  # Then 15-minute chunks
        self.overlap_duration = 60  # 60 seconds overlap
        self.chunk_count = 0

        # Dynamic buffer size based on chunk count
        self.initial_buffer_size = int(self.sample_rate * self.initial_chunk_duration)
        self.full_buffer_size = int(self.sample_rate * self.full_chunk_duration)
        self.overlap_size = int(self.sample_rate * self.overlap_duration)

        # Transcript storage
        self.text_buffer = []

        # Speaker consistency tracking - Initialize early
        self.person_counter = 1
        self.global_speaker_mapping = {}  # Maps diarization labels to Person X
        self.chunk_counter = 0

        # --- Model Initialization ---
        print("Initializing ASR + Diarization pipeline...")
        try:
            with open(os.devnull, 'w') as devnull, \
                 contextlib.redirect_stdout(devnull), \
                 contextlib.redirect_stderr(devnull):

                # Initialize ASR pipeline with MPS acceleration
                device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
                torch_dtype = torch.float32  # Use float32 for better MPS compatibility

                self.asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    torch_dtype=torch_dtype,
                    device=device,
                    return_timestamps=True,
                    generate_kwargs={"max_new_tokens": 448}  # Use max_new_tokens instead of deprecated max_length
                )

                # Test MPS compatibility with a small sample
                if device == "mps":
                    try:
                        # Test MPS with a small audio sample
                        test_audio = np.random.randn(1000).astype(np.float32)
                        _ = self.asr_pipeline(test_audio, return_timestamps=True)
                        print(f"✅ MPS acceleration working correctly")
                    except Exception as mps_error:
                        print(f"⚠️ MPS compatibility issue detected, falling back to CPU")
                        print(f"MPS Error: {str(mps_error)[:100]}...")

                        # Reinitialize with CPU
                        device = "cpu"
                        self.asr_pipeline = pipeline(
                            "automatic-speech-recognition",
                            model=model_name,
                            torch_dtype=torch.float32,
                            device="cpu",
                            return_timestamps=True,
                            generate_kwargs={"max_new_tokens": 448}
                        )

                # Initialize Diarization pipeline with MPS acceleration
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )

                if device == "mps":
                    try:
                        self.diarization_pipeline = self.diarization_pipeline.to(torch.device("mps"))
                        print(f"✅ Diarization using MPS acceleration")
                    except Exception as mps_error:
                        print(f"⚠️ Diarization MPS issue, using CPU")
                        # Diarization will use CPU by default
                elif device == "cuda":
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))

                # Initialize summarization pipeline
                self.summarizer = pipeline("summarization", model="knkarthick/meeting-summary-samsum")

                # Store device info for display
                self.device_info = device

            print(f"Models loaded successfully on device: {self.device_info}\n")
        except Exception as e:
            print(f"\nError during model initialization: {e}")
            sys.exit(1)

        # File output
        self.output_filename = None

    def process_audio_chunk(self, audio_data):
        """Process audio chunk with integrated ASR + Diarization"""
        try:
            # Save audio to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, self.sample_rate)
                temp_filename = temp_file.name

            try:
                # Run diarization
                if self.debug:
                    print(f"\n🎯 Running diarization on {len(audio_data)/self.sample_rate:.1f}s audio chunk...")

                diarization = self.diarization_pipeline(temp_filename)

                # Run ASR
                if self.debug:
                    print(f"🎤 Running ASR transcription...")

                asr_result = self.asr_pipeline(
                    audio_data,
                    return_timestamps=True
                )

                # Combine ASR and diarization results
                segments = self.align_asr_with_diarization(asr_result, diarization, temp_filename)

                # Increment chunk counter for speaker consistency tracking
                self.increment_chunk_counter()

                return segments

            finally:
                # Clean up temporary file
                os.unlink(temp_filename)

        except Exception as e:
            print(f"Error processing audio chunk: {str(e)}", file=sys.stderr)
            return []

    def align_asr_with_diarization(self, asr_result, diarization, audio_file):
        """Align ASR chunks with speaker diarization and maintain speaker consistency"""
        segments = []

        # Get ASR chunks with timestamps
        asr_chunks = asr_result.get("chunks", [])
        if not asr_chunks:
            return segments

        # Create speaker timeline
        speaker_timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_timeline.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })

        # Track speakers in this chunk for voice switch detection
        chunk_speakers = set([s['speaker'] for s in speaker_timeline])
        self.current_chunk_speakers = chunk_speakers

        if self.debug:
            print(f"📊 Found {len(speaker_timeline)} speaker segments")
            print(f"📝 Found {len(asr_chunks)} ASR chunks")
            print(f"🎭 Unique speakers in chunk: {chunk_speakers}")

        # Assign speakers to ASR chunks
        for chunk in asr_chunks:
            chunk_start = chunk['timestamp'][0] if chunk['timestamp'][0] is not None else 0
            chunk_end = chunk['timestamp'][1] if chunk['timestamp'][1] is not None else chunk_start + 1
            chunk_text = chunk['text'].strip()

            if not chunk_text:
                continue

            # Find best matching speaker
            best_speaker = self.find_best_speaker(chunk_start, chunk_end, speaker_timeline)

            if best_speaker:
                # Map speaker labels to Person X format with consistency check
                person_name = self.get_consistent_person_name(best_speaker, chunk_start, chunk_end, audio_file)

                segments.append({
                    'speaker': person_name,
                    'text': chunk_text,
                    'start_time': chunk_start,
                    'end_time': chunk_end
                })

                if self.debug:
                    print(f"🎯 [{chunk_start:.1f}s-{chunk_end:.1f}s] {best_speaker} → {person_name}: {chunk_text[:50]}...")

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

    def get_consistent_person_name(self, speaker_label, start_time, end_time, audio_file):
        """Convert speaker labels to Person X format with cross-chunk consistency"""
        # Simplified and more reliable approach
        # Use a simple persistent mapping from diarization labels to Person names
        
        if speaker_label not in self.global_speaker_mapping:
            # Create new person for this speaker label
            person_name = f"Person {self.person_counter}"
            self.global_speaker_mapping[speaker_label] = person_name
            self.person_counter += 1
            print(f"\n🎤 New speaker identified: {speaker_label} → {person_name}")
            if self.debug:
                print(f"🆕 Mapped {speaker_label} to {person_name}")
        else:
            person_name = self.global_speaker_mapping[speaker_label]
            if self.debug:
                print(f"✅ {speaker_label} continues as {person_name}")
        
        return person_name

    def detect_voice_switch(self, speaker_label, existing_person, current_chunk):
        """Detect if the same speaker label now represents a different voice"""
        # Be more conservative - only detect voice switches in specific scenarios

        # For now, disable automatic voice switch detection as it's too aggressive
        # The simple mapping approach should work better

        # Only detect voice switches in very obvious cases:
        # 1. If we see SPEAKER_00 in a chunk where we previously saw SPEAKER_01 as the main speaker
        # 2. And the previous chunk had clear speaker separation

        if self.debug:
            print(f"🔍 Voice switch check: {speaker_label} → {existing_person} (chunk {current_chunk})")

        # For now, return False to disable aggressive voice switching
        # This will rely on the diarization model being more accurate
        return False

    def get_current_chunk_speakers(self):
        """Get list of speakers in current chunk"""
        # This would be populated during chunk processing
        # For now, return empty list as fallback
        return []

    def increment_chunk_counter(self):
        """Increment chunk counter for tracking"""
        if hasattr(self, 'chunk_counter'):
            self.chunk_counter += 1
        else:
            self.chunk_counter = 0

    def get_person_name_fallback(self, speaker_label):
        """Fallback method for speaker mapping without embeddings"""
        if not hasattr(self, 'speaker_mapping'):
            self.speaker_mapping = {}

        if speaker_label not in self.speaker_mapping:
            person_name = f"Person {self.person_counter}"
            self.speaker_mapping[speaker_label] = person_name
            self.person_counter += 1
            print(f"\n🎤 New speaker identified, creating {person_name}")

        return self.speaker_mapping[speaker_label]

    def process_audio(self):
        """Main audio processing loop"""
        while self.is_recording:
            try:
                audio_data = self.audio_queue.get(timeout=1)

                # Skip if we're stopping
                if self.is_stopping:
                    break

                # Process the audio chunk
                segments = self.process_audio_chunk(audio_data)

                # Display and store results
                if segments:
                    for segment in segments:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"\n[{timestamp}] 👤 {segment['speaker']}: {segment['text']}")
                        self.text_buffer.append(f"{segment['speaker']}: {segment['text']}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error during transcription: {str(e)}", file=sys.stderr)

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input"""
        if status:
            print(f"Status: {status}", file=sys.stderr)

        if self.is_recording:
            self.audio_buffer.extend(indata.flatten())

            # Show periodic status updates during long recording periods
            buffer_seconds = len(self.audio_buffer) / self.sample_rate
            if hasattr(self, 'last_status_time'):
                if buffer_seconds - self.last_status_time >= 30:  # Every 30 seconds
                    minutes = int(buffer_seconds // 60)
                    seconds = int(buffer_seconds % 60)
                    print(f"🎙️  Recording... {minutes:02d}:{seconds:02d} / 15:00")
                    self.last_status_time = buffer_seconds
            else:
                self.last_status_time = 0

            # Use dynamic buffer size: short chunks initially, then long chunks
            current_buffer_size = self.initial_buffer_size if self.chunk_count < 3 else self.full_buffer_size

            if len(self.audio_buffer) >= current_buffer_size:
                audio_data = np.array(self.audio_buffer[:current_buffer_size]).astype(np.float32)

                # Keep overlap for next chunk (but only for longer chunks)
                if self.chunk_count >= 3:
                    overlap_start = current_buffer_size - self.overlap_size
                    self.audio_buffer = self.audio_buffer[overlap_start:]
                else:
                    # For initial short chunks, no overlap needed
                    self.audio_buffer = self.audio_buffer[current_buffer_size:]

                # Normalize audio
                if np.max(np.abs(audio_data)) > 0:
                    audio_data /= np.max(np.abs(audio_data))

                chunk_duration = current_buffer_size / self.sample_rate
                if self.chunk_count < 3:
                    print(f"\n🎯 Processing {chunk_duration:.0f}-second chunk (quick start)...")
                else:
                    print(f"\n🎯 Processing {chunk_duration/60:.1f}-minute chunk...")

                self.audio_queue.put(audio_data)
                self.chunk_count += 1

                # Reset status timer
                self.last_status_time = 0

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
        """Start the recording and transcription process"""
        self.is_recording = True

        print("Starting audio stream...")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * 0.5)
        )

        self.stream.start()

        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()

        print("\n🎙️  Recording started. Press Ctrl+C to stop.\n")

    def stop_recording(self):
        """Stop the recording and transcription process"""
        if self.is_stopping:
            return
        self.is_stopping = True

        print("\nStopping recording...")
        self.is_recording = False

        # Process any remaining audio in the buffer before stopping
        if hasattr(self, 'audio_buffer') and len(self.audio_buffer) > 0:
            buffer_seconds = len(self.audio_buffer) / self.sample_rate
            if buffer_seconds > 5:  # Only process if we have at least 5 seconds
                print(f"🎯 Processing final {buffer_seconds:.1f}s of audio...")

                # Create final chunk from remaining buffer
                final_audio = np.array(self.audio_buffer).astype(np.float32)

                # Normalize audio
                if np.max(np.abs(final_audio)) > 0:
                    final_audio /= np.max(np.abs(final_audio))

                # Process the final chunk
                try:
                    segments = self.process_audio_chunk(final_audio)

                    # Display results immediately
                    if segments:
                        for segment in segments:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"\n[{timestamp}] 👤 {segment['speaker']}: {segment['text']}")
                            self.text_buffer.append(f"{segment['speaker']}: {segment['text']}")
                except Exception as e:
                    print(f"Warning: Error processing final audio chunk: {e}")

        try:
            # Stop the processing thread
            if hasattr(self, 'process_thread'):
                self.process_thread.join(timeout=3.0)
                if self.process_thread.is_alive():
                    print("Warning: Processing thread did not stop cleanly")

            # Stop the audio stream
            if hasattr(self, 'stream') and self.stream.active:
                self.stream.stop()
                self.stream.close()

            # Drain any remaining items from the audio queue
            try:
                while not self.audio_queue.empty():
                    remaining_audio = self.audio_queue.get_nowait()
                    print(f"🎯 Processing remaining queued audio...")
                    segments = self.process_audio_chunk(remaining_audio)

                    if segments:
                        for segment in segments:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"\n[{timestamp}] 👤 {segment['speaker']}: {segment['text']}")
                            self.text_buffer.append(f"{segment['speaker']}: {segment['text']}")
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Warning: Error processing queued audio: {e}")

        except Exception as e:
            print(f"Warning: Error stopping audio components: {str(e)}")

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
                filename = self.save_meeting_to_file("Summary generation failed due to interruption.")
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
    print(f"⏱️  Chunking: 30s (quick start) → 15min (accuracy)")
    print(f"🚀  Processing: Real-time with MPS/GPU acceleration")
    
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