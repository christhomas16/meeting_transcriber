import warnings
warnings.filterwarnings("ignore")

import sounddevice as sd
import numpy as np
import whisper
import torch
from transformers import pipeline
import threading
import queue
import time
import sys
from datetime import datetime
from pyannote.audio import Pipeline, Model
from pyannote.core import Annotation
import os
from dotenv import load_dotenv
import logging
import re
import contextlib
from tqdm import tqdm

# Suppress all logging messages
logging.basicConfig(level=logging.CRITICAL)

# Globally disable tqdm progress bars
tqdm.disable = True
tqdm.__init__ = lambda self, *args, **kwargs: None

class SpeakerIdentifier:
    def __init__(self):
        self.people = {}  # person_name -> voice_characteristics and history
        self.unknown_speakers = {}  # SPEAKER_XX -> person_id for unknown speakers
        self.person_counter = 1
        self.voice_history = []  # Track all voice samples with their assigned person
        
    def extract_name_from_text(self, text):
        """Extract potential names from text like 'my name is John' or 'I'm Anna'"""
        text_lower = text.lower().strip()
        
        # More specific patterns with better context
        patterns = [
            r"my name is (\w+)(?:\.|,|\s|$)",  # "my name is Chris" (end or punctuation)
            r"(?:^|\s)call me (\w+)(?:\.|,|\s|$)",  # "call me Anna"
            r"(?:^|\s)this is (\w+)(?:\.|,|\s|$)",  # "this is Mike"
        ]
        
        # Special handling for "I'm" pattern with stricter context
        im_match = re.search(r"(?:^|\s)i'?m (\w+)", text_lower)
        if im_match:
            potential_name = im_match.group(1).capitalize()
            
            # Check if it's followed by context that suggests it's NOT a name
            following_text = text_lower[im_match.end():im_match.end()+20]
            
            # If followed by verbs/adjectives, it's probably not a name
            non_name_contexts = [
                'going', 'doing', 'trying', 'working', 'playing', 'thinking', 'feeling',
                'happy', 'sad', 'good', 'bad', 'fine', 'okay', 'ready', 'here', 'there',
                'just', 'still', 'always', 'never', 'really', 'very', 'quite', 'pretty',
                'chilling', 'loving', 'enjoying', 'wondering', 'looking', 'hoping',
                'excited', 'tired', 'busy', 'free', 'available', 'sorry', 'glad'
            ]
            
            # Check if the word after "I'm" suggests it's not a name
            if potential_name.lower() in non_name_contexts:
                pass  # Skip this potential name
            else:
                # Additional check: if it's a real name, add it to consideration
                patterns.append(r"(?:^|\s)i'?m (\w+)(?:\.|,|\s|$)")
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).capitalize()
                
                # Comprehensive list of words that are definitely NOT names
                excluded_words = {
                    # Common verbs (present participle and base form)
                    'going', 'doing', 'saying', 'calling', 'trying', 'working', 'playing',
                    'thinking', 'feeling', 'looking', 'hoping', 'wanting', 'getting',
                    'making', 'taking', 'giving', 'coming', 'leaving', 'staying',
                    'chilling', 'loving', 'enjoying', 'wondering', 'excited', 'tired',
                    
                    # Common adjectives/states
                    'good', 'bad', 'okay', 'fine', 'great', 'nice', 'cool', 'awesome',
                    'happy', 'sad', 'ready', 'busy', 'free', 'available', 'sorry', 'glad',
                    'still', 'just', 'really', 'very', 'quite', 'pretty', 'always', 'never',
                    
                    # Articles, conjunctions, prepositions
                    'the', 'a', 'an', 'and', 'or', 'but', 'so', 'if', 'when', 'where',
                    'here', 'there', 'now', 'then', 'today', 'tomorrow', 'yesterday',
                    
                    # Common conversational words
                    'actually', 'basically', 'literally', 'totally', 'definitely',
                    'probably', 'maybe', 'perhaps', 'certainly', 'absolutely',
                    
                    # Single letters that might be misheard
                    'a', 'i', 'o', 'u', 'e'
                }
                
                # Only accept if:
                # 1. Not in excluded words
                # 2. At least 2 characters
                # 3. Only alphabetic
                # 4. Starts with capital letter (proper noun)
                if (len(name) >= 2 and 
                    name.lower() not in excluded_words and
                    name.isalpha() and
                    name[0].isupper()):
                    return name
        return None
    
    def get_voice_features(self, audio_segment):
        """Extract voice characteristics for comparison"""
        try:
            if len(audio_segment) < 1000:  # Too short
                return None
                
            # Basic statistics
            amplitude_mean = np.mean(np.abs(audio_segment))
            amplitude_std = np.std(audio_segment)
            amplitude_max = np.max(np.abs(audio_segment))
            
            # Frequency domain features
            fft = np.fft.fft(audio_segment)
            freqs = np.fft.fftfreq(len(audio_segment), 1/16000)
            magnitude = np.abs(fft)
            
            # Focus on speech frequencies (80-8000 Hz)
            speech_indices = (freqs >= 80) & (freqs <= 8000)
            if np.sum(speech_indices) == 0:
                return None
                
            speech_mag = magnitude[speech_indices]
            speech_freqs = freqs[speech_indices]
            
            # Spectral centroid (average frequency)
            if np.sum(speech_mag) > 0:
                spectral_centroid = np.sum(speech_freqs * speech_mag) / np.sum(speech_mag)
            else:
                spectral_centroid = 1000
                
            # Energy distribution in frequency bands
            low_freq = (speech_freqs >= 80) & (speech_freqs <= 300)
            mid_freq = (speech_freqs >= 300) & (speech_freqs <= 2000)
            high_freq = (speech_freqs >= 2000) & (speech_freqs <= 8000)
            
            total_energy = np.sum(speech_mag)
            if total_energy > 0:
                low_energy_ratio = np.sum(speech_mag[low_freq]) / total_energy
                mid_energy_ratio = np.sum(speech_mag[mid_freq]) / total_energy
                high_energy_ratio = np.sum(speech_mag[high_freq]) / total_energy
            else:
                low_energy_ratio = mid_energy_ratio = high_energy_ratio = 0.33
            
            return {
                'amplitude_mean': amplitude_mean,
                'amplitude_std': amplitude_std,
                'amplitude_max': amplitude_max,
                'spectral_centroid': abs(spectral_centroid),
                'low_energy_ratio': low_energy_ratio,
                'mid_energy_ratio': mid_energy_ratio,
                'high_energy_ratio': high_energy_ratio
            }
        except:
            return None
    
    def compare_voices(self, features1, features2):
        """Compare two voice feature sets and return similarity (0-1)"""
        if not features1 or not features2:
            return 0.0
        
        try:
            # Normalize features
            amp_sim = 1 - abs(features1['amplitude_mean'] - features2['amplitude_mean']) / max(features1['amplitude_mean'], features2['amplitude_mean'], 0.001)
            freq_sim = 1 - abs(features1['spectral_centroid'] - features2['spectral_centroid']) / 4000
            
            # Energy distribution similarity
            low_sim = 1 - abs(features1['low_energy_ratio'] - features2['low_energy_ratio'])
            mid_sim = 1 - abs(features1['mid_energy_ratio'] - features2['mid_energy_ratio'])
            high_sim = 1 - abs(features1['high_energy_ratio'] - features2['high_energy_ratio'])
            
            # Weighted average
            similarity = (0.2 * amp_sim + 0.3 * freq_sim + 0.2 * low_sim + 0.2 * mid_sim + 0.1 * high_sim)
            return max(0, min(1, similarity))
        except:
            return 0.0
    
    def identify_speaker(self, speaker_label, audio_segment, text=""):
        """Identify who is speaking using voice matching and content analysis"""
        
        voice_features = self.get_voice_features(audio_segment)
        
        # Check if someone introduces themselves
        introduced_name = self.extract_name_from_text(text)
        if introduced_name:
            # Debug: Show when name is detected
            print(f"🎯 Detected name introduction: '{introduced_name}' from text: '{text[:50]}...'")
            
            # AGGRESSIVE EARLY DETECTION: If this is a new name, immediately create a profile
            if introduced_name not in self.people:
                # New person introducing themselves
                self.people[introduced_name] = {
                    'voice_samples': [voice_features] if voice_features else [],
                    'speaker_labels_seen': [speaker_label],
                    'introduction_count': 1
                }
                
                # COLD START FIX: Check if any existing unknown speakers should be reassigned
                # Look for recent unknown speaker assignments with this speaker_label
                speakers_to_reassign = []
                for i, assignment in enumerate(self.voice_history[-5:]):  # Check last 5 assignments
                    if (assignment['speaker_label'] == speaker_label and 
                        assignment['person'].startswith('Person ') and
                        assignment['confidence'] < 0.5):  # Low confidence unknown assignment
                        speakers_to_reassign.append(len(self.voice_history) - 5 + i)
                
                # Reassign those segments to the newly introduced person
                for idx in speakers_to_reassign:
                    self.voice_history[idx]['person'] = introduced_name
                    self.voice_history[idx]['confidence'] = 0.8  # Higher confidence now
                
            else:
                # Known person, add new voice sample
                if voice_features and len(self.people[introduced_name]['voice_samples']) < 5:
                    self.people[introduced_name]['voice_samples'].append(voice_features)
                if speaker_label not in self.people[introduced_name]['speaker_labels_seen']:
                    self.people[introduced_name]['speaker_labels_seen'].append(speaker_label)
                self.people[introduced_name]['introduction_count'] += 1
            
            # Record this voice assignment
            self.voice_history.append({
                'voice_features': voice_features,
                'person': introduced_name,
                'speaker_label': speaker_label,
                'confidence': 1.0  # High confidence for self-introduction
            })
            return introduced_name
        
        # No self-introduction, try to match voice with known people
        if voice_features and self.people:
            best_match = None
            best_score = 0
            
            # Compare with known people
            for person_name, person_data in self.people.items():
                if not person_data['voice_samples']:
                    continue
                    
                # Calculate similarity with voice samples
                similarities = []
                for stored_voice in person_data['voice_samples'][-3:]:  # Use recent samples
                    sim = self.compare_voices(voice_features, stored_voice)
                    similarities.append(sim)
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    max_similarity = np.max(similarities)
                    
                    # Use the better of average or max similarity
                    voice_score = max(avg_similarity, max_similarity * 0.8)
                    
                    # Strong bonus for speaker label consistency
                    label_bonus = 0.15 if speaker_label in person_data['speaker_labels_seen'] else 0
                    
                    total_score = voice_score + label_bonus
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_match = person_name
            
            # Adaptive threshold: Be more conservative when we already have multiple speakers
            total_voice_samples = sum(len(data['voice_samples']) for data in self.people.values())
            num_known_people = len(self.people)
            
            if num_known_people == 0:  # No speakers yet
                threshold = 0.3  # Aggressive for first speaker
            elif num_known_people == 1:  # One speaker known
                threshold = 0.35  # Slightly conservative for second speaker
            elif total_voice_samples < 6:  # Early learning phase
                threshold = 0.45  # More conservative for third+ speakers
            else:
                threshold = 0.5  # High threshold when established (prevent false matches)
                
            if best_match and best_score > threshold:
                # Debug: Show voice matching
                print(f"🔗 Voice matched: {speaker_label} → {best_match} (score: {best_score:.3f}, threshold: {threshold:.3f})")
                
                # Add this voice sample to the matched person
                if len(self.people[best_match]['voice_samples']) < 5:
                    self.people[best_match]['voice_samples'].append(voice_features)
                if speaker_label not in self.people[best_match]['speaker_labels_seen']:
                    self.people[best_match]['speaker_labels_seen'].append(speaker_label)
                
                # Record this voice assignment
                self.voice_history.append({
                    'voice_features': voice_features,
                    'person': best_match,
                    'speaker_label': speaker_label,
                    'confidence': best_score
                })
                return best_match
            else:
                # Debug: Show when voice matching fails
                if best_match:
                    print(f"❌ Voice match failed: {speaker_label} → {best_match} (score: {best_score:.3f} < threshold: {threshold:.3f})")
                else:
                    print(f"❌ No voice matches found for {speaker_label}")
        
        # Conservative fallback: only reassign to known people if there's strong evidence
        if self.people and speaker_label.startswith('SPEAKER_'):
            known_people = list(self.people.keys())
            
            # Check if this speaker_label is strongly associated with one person
            label_counts = {}
            for person_name in known_people:
                count = self.people[person_name]['speaker_labels_seen'].count(speaker_label)
                if count > 0:
                    label_counts[person_name] = count
            
            # If only one person has been seen with this label AND they've used it multiple times, use them
            if len(label_counts) == 1:
                person_name, count = list(label_counts.items())[0]
                if count >= 2:  # Require multiple uses for confidence
                    if speaker_label not in self.people[person_name]['speaker_labels_seen']:
                        self.people[person_name]['speaker_labels_seen'].append(speaker_label)
                    return person_name
            
            # If multiple people have used this label, don't guess - create new person
            elif len(label_counts) > 1:
                pass  # Fall through to create new person
            
            # If no one has been seen with this label, be conservative:
            # Only assign to existing person if we have very few speakers (≤2)
            elif len(label_counts) == 0 and len(self.people) <= 2:
                person_label_counts = [(name, len(data['speaker_labels_seen'])) for name, data in self.people.items()]
                person_label_counts.sort(key=lambda x: x[1])  # Sort by label count
                best_person = person_label_counts[0][0]  # Person with fewest labels
                self.people[best_person]['speaker_labels_seen'].append(speaker_label)
                return best_person
        
        # Standard fallback: use speaker labels for unknown speakers
        if speaker_label in self.unknown_speakers:
            return self.unknown_speakers[speaker_label]
        
        # Create new unknown speaker
        person_id = f"Person {self.person_counter}"
        self.person_counter += 1
        self.unknown_speakers[speaker_label] = person_id
        
        # Debug: Show when new speaker is created
        print(f"🆕 Created new speaker: {person_id} for {speaker_label}")
        
        # Record this voice assignment
        if voice_features:
            self.voice_history.append({
                'voice_features': voice_features,
                'person': person_id,
                'speaker_label': speaker_label,
                'confidence': 0.3  # Low confidence for unknown
            })
        
        return person_id

class MeetingTranscriber:
    def __init__(self, max_speakers_limit=15):
        """
        Initialize the meeting transcriber.
        
        Args:
            max_speakers_limit (int): Maximum number of speakers to detect. 
                                    Higher numbers allow more speakers but may reduce accuracy.
                                    Recommended: 5-10 for best performance, 15+ for large groups.
        """
        # Load environment variables and check for token
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("\nError: Hugging Face token not found. Please see README.md for setup instructions.")
            sys.exit(1)
        
        # Speaker configuration
        self.max_speakers_limit = max_speakers_limit
        
        # Audio parameters
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.is_stopping = False
        
        # Audio buffer
        self.audio_buffer = []
        self.buffer_duration = 10
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        
        # Speaker identification
        self.speaker_identifier = SpeakerIdentifier()
        
        # Transcript correction tracking
        self.recent_segments = []  # Track recent segments for potential correction
        
        # --- Model Initialization ---
        print("Initializing models...")
        try:
            with open(os.devnull, 'w') as devnull, \
                 contextlib.redirect_stdout(devnull), \
                 contextlib.redirect_stderr(devnull):
                
                self.whisper_model = whisper.load_model("base")
                
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )

                self.embedding_model = Model.from_pretrained(
                    "pyannote/embedding",
                    use_auth_token=hf_token
                )
                
                self.summarizer = pipeline("summarization", model="knkarthick/meeting-summary-samsum")
            
            print("Models loaded successfully.\n")
        except Exception as e:
            print(f"\nError during model initialization: {e}")
            sys.exit(1)
        
        # Summarization parameters
        self.text_buffer = []
        self.min_text_length = 30

    def process_audio(self):
        """Processes audio by aligning speaker diarization with word-level timestamps."""
        while self.is_recording:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                
                with open(os.devnull, 'w') as devnull, \
                     contextlib.redirect_stdout(devnull), \
                     contextlib.redirect_stderr(devnull):
                    
                    waveform = torch.from_numpy(audio_data).unsqueeze(0)
                    
                    # Adaptive speaker detection - start conservative, expand as more speakers are detected
                    known_speakers = len(self.speaker_identifier.people) + len(self.speaker_identifier.unknown_speakers)
                    max_speakers_to_try = min(self.max_speakers_limit, max(3, known_speakers + 2))
                    
                    diarization = self.diarization_pipeline(
                        {"waveform": waveform, "sample_rate": self.sample_rate},
                        min_speakers=1,
                        max_speakers=max_speakers_to_try
                    )

                    transcription_result = self.whisper_model.transcribe(audio_data, word_timestamps=True, language="en")

                full_transcript = self.align_transcription_with_diarization(diarization, transcription_result, audio_data)
                
                if full_transcript:
                    for segment in full_transcript:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        
                        # Store segment for potential correction
                        segment_info = {
                            'timestamp': timestamp,
                            'speaker': segment['speaker'],
                            'text': segment['text'],
                            'corrected': False
                        }
                        self.recent_segments.append(segment_info)
                        
                        # Check if we should correct any recent segments based on new speaker info
                        self._check_for_corrections()
                        
                        print(f"\n[{timestamp}] 👤 {segment['speaker']}: {segment['text']}")
                        self.text_buffer.append(f"{segment['speaker']}: {segment['text']}")
                        
                    # Keep only last 10 segments for correction purposes
                    if len(self.recent_segments) > 10:
                        self.recent_segments = self.recent_segments[-10:]

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error during transcription: {str(e)}", file=sys.stderr)

    def align_transcription_with_diarization(self, diarization: Annotation, transcription_result: dict, audio_data: np.ndarray):
        """Simplified approach: match Whisper segments directly to speakers."""
        
        # Create speaker segments from diarization
        speaker_segments = {}
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if turn.end - turn.start >= 0.3:  # Minimum duration
                start_sample = int(turn.start * self.sample_rate)
                end_sample = int(turn.end * self.sample_rate)
                segment_audio = audio_data[start_sample:end_sample]
                
                speaker_segments[speaker_label] = {
                    'turn': turn,
                    'audio': segment_audio,
                    'duration': turn.end - turn.start
                }
        
        # Get Whisper segments (these are natural speech segments)
        whisper_segments = transcription_result.get("segments", [])
        if not whisper_segments:
            return []
        
        final_results = []
        
        for segment in whisper_segments:
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", segment_start + 1)
            segment_text = segment.get("text", "").strip()
            
            if not segment_text:
                continue
            
            # Find the speaker with the best time overlap
            best_speaker_label = None
            best_overlap = 0
            best_overlap_ratio = 0
            
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                # Calculate time overlap
                overlap_start = max(segment_start, turn.start)
                overlap_end = min(segment_end, turn.end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # Calculate overlap ratio (what percentage of the segment is covered)
                segment_duration = segment_end - segment_start
                if segment_duration > 0:
                    overlap_ratio = overlap_duration / segment_duration
                else:
                    overlap_ratio = 0
                
                # Use the speaker with the best overlap (prefer high overlap ratio)
                if overlap_ratio > best_overlap_ratio or (overlap_ratio == best_overlap_ratio and overlap_duration > best_overlap):
                    best_overlap = overlap_duration
                    best_overlap_ratio = overlap_ratio
                    best_speaker_label = speaker_label
            
            # Only process if we found a reasonable match and have audio for this speaker
            if best_speaker_label and best_speaker_label in speaker_segments and best_overlap_ratio > 0.1:
                # Identify the speaker using voice characteristics
                person_id = self.speaker_identifier.identify_speaker(
                    best_speaker_label,
                    speaker_segments[best_speaker_label]['audio'],
                    segment_text
                )
                
                final_results.append({
                    'speaker': person_id,
                    'text': segment_text,
                    'start_time': segment_start
                })
        
        # Sort by start time
        final_results.sort(key=lambda x: x["start_time"])
        return final_results

    def clean_text(self, text):
        """Clean transcribed text"""
        # Remove extra whitespace and common transcription artifacts
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\b(um|uh|hmm|ah)\b', '', text, flags=re.IGNORECASE)
        text = text.strip()
        return text if text else None

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input"""
        if status:
            print(f"Status: {status}", file=sys.stderr)
        
        if self.is_recording:
            self.audio_buffer.extend(indata.flatten())
            
            if len(self.audio_buffer) >= self.buffer_size:
                audio_data = np.array(self.audio_buffer[:self.buffer_size]).astype(np.float32)
                self.audio_buffer = self.audio_buffer[self.buffer_size:]
                
                # Normalize audio
                if np.max(np.abs(audio_data)) > 0:
                    audio_data /= np.max(np.abs(audio_data))
                
                self.audio_queue.put(audio_data)

    def generate_final_summary(self):
        """Generate a summary of the entire conversation"""
        if not self.text_buffer:
            return
            
        try:
            print("\n📝 Generating final summary...")
            combined_text = " ".join(self.text_buffer)
            
            if len(combined_text) > self.min_text_length:
                # Calculate appropriate summary length
                min_len = 20
                max_len = max(min_len + 1, min(150, len(combined_text) // 4))
                
                summary = self.summarizer(combined_text, max_length=max_len, min_length=min_len, do_sample=False)
                summary_text = summary[0]['summary_text']
                print("\n=== Final Summary ===")
                print(summary_text)
                print("=====================\n")
            else:
                print("\n=== Conversation Too Short for Summary ===")
                print("Here's the full conversation:")
                for line in self.text_buffer:
                    print(f"  {line}")
                print("==========================================\n")
        except Exception as e:
            print(f"Error during final summarization: {str(e)}", file=sys.stderr)
            
    def stop_recording(self):
        """Stop the recording and transcription process"""
        if self.is_stopping:
            return
        self.is_stopping = True

        print("\nStopping recording...")
        self.is_recording = False
        
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2.0)
        
        if hasattr(self, 'stream') and self.stream.active:
            self.stream.stop()
            self.stream.close()
        
        self.generate_final_summary()
        
        print("\nRecording stopped. Goodbye!")
        
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
    
    def _check_for_corrections(self):
        """Check if recent speaker identifications should be corrected based on new information"""
        if len(self.recent_segments) < 2:
            return
            
        # Look for cases where someone introduces themselves and previous segments should be corrected
        latest_segment = self.recent_segments[-1]
        
        # If latest segment contains a name introduction, check if recent segments should be reassigned
        if any(pattern in latest_segment['text'].lower() for pattern in ['my name is', "i'm ", 'i am ']):
            speaker_name = latest_segment['speaker']
            
            # Only correct if this is not a "Person X" label (meaning it's a real name)
            if not speaker_name.startswith('Person '):
                # Look back at recent segments for potential misattributions
                for i in range(len(self.recent_segments) - 2, max(-1, len(self.recent_segments) - 6), -1):
                    past_segment = self.recent_segments[i]
                    
                    # If a past segment was attributed to a "Person X" but contained this person's voice patterns
                    if (past_segment['speaker'].startswith('Person ') and 
                        not past_segment['corrected'] and
                        self._should_correct_attribution(past_segment, speaker_name)):
                        
                        # Print correction notice
                        print(f"\n🔄 CORRECTION: [{past_segment['timestamp']}] was actually {speaker_name}")
                        
                        # Update the segment
                        past_segment['speaker'] = speaker_name
                        past_segment['corrected'] = True
                        
                        # Update text buffer (find and replace the entry)
                        old_entry = f"Person {past_segment['speaker'].split()[-1]}: {past_segment['text']}"
                        new_entry = f"{speaker_name}: {past_segment['text']}"
                        
                        for j, buffer_item in enumerate(self.text_buffer):
                            if past_segment['text'] in buffer_item and old_entry.replace(f"Person {past_segment['speaker'].split()[-1]}", "Person") in buffer_item:
                                self.text_buffer[j] = new_entry
                                break
    
    def _should_correct_attribution(self, past_segment, current_speaker):
        """Determine if a past segment should be corrected to the current speaker"""
        # Simple heuristic: if the past segment was recent (within last 3 segments) 
        # and was assigned to an unknown person, it might be worth correcting
        
        # Look for content clues that suggest it's the same person
        past_text = past_segment['text'].lower()
        
        # If past segment contained conversational continuity or similar speech patterns
        if any(word in past_text for word in ['i', 'me', 'my', 'am', 'was']):
            return True
            
        return False

def main():
    # You can customize the maximum speaker limit here
    # Examples:
    # transcriber = MeetingTranscriber(max_speakers_limit=5)   # Small meetings
    # transcriber = MeetingTranscriber(max_speakers_limit=10)  # Medium groups  
    # transcriber = MeetingTranscriber(max_speakers_limit=20)  # Large conferences
    
    transcriber = MeetingTranscriber(max_speakers_limit=15)  # Default: up to 15 speakers
    
    print(f"🎙️  Configured for up to {transcriber.max_speakers_limit} speakers")
    
    try:
        transcriber.start_recording()
        while transcriber.is_recording:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C received, shutting down...")
    finally:
        transcriber.stop_recording()

if __name__ == "__main__":
    main() 