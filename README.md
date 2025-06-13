# Real-Time Meeting Transcriber with Integrated ASR + Diarization

A sophisticated real-time meeting transcriber that uses an integrated ASR + Diarization pipeline for accurate speaker identification and transcription. Built using the [Hugging Face ASR + Diarization approach](https://huggingface.co/blog/asr-diarization).

## ✨ Features

- **🎙️ Real-Time Transcription**: Live speech-to-text using OpenAI Whisper
- **🎯 Integrated Diarization**: Uses pyannote speaker-diarization-3.1 for accurate speaker detection
- **👥 Automatic Speaker Labeling**: Clean "Person 1", "Person 2" labels without complex voice analysis
- **⚡ Optimized Pipeline**: ASR and diarization work together in a single optimized workflow
- **📝 Timestamp Alignment**: Precise alignment of transcription with speaker segments
- **📊 Final Summary**: Comprehensive conversation summary when you end the session
- **🔄 Continuous Processing**: 10-second audio chunks for optimal real-time performance

## 🚀 How It Works

The system uses an integrated pipeline that combines Whisper ASR with pyannote diarization:

1. **Audio Capture**: Records from your microphone in 10-second chunks
2. **Diarization**: Uses pyannote/speaker-diarization-3.1 to identify speaker segments with precise timestamps
3. **ASR Transcription**: Converts audio to text using Whisper with timestamps
4. **Intelligent Alignment**: Matches transcribed text chunks to speaker segments based on time overlap
5. **Consistent Labeling**: Maintains "Person 1", "Person 2" labels across all audio chunks
6. **Final Summarization**: Generates conversation summary using specialized meeting summary model

## 🛠️ Installation

### Prerequisites
- Python 3.8-3.11 (Python 3.13 is not yet supported)
- Microphone access
- Internet connection (for model downloads)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd meeting-transcriber

# Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2. Hugging Face Configuration
**⚠️ Critical Step**: The diarization pipeline requires Hugging Face access.

1. **Create Account & Token**:
   - Sign up at [huggingface.co](https://huggingface.co)
   - Generate token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

2. **Accept Model Terms** (Required):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

3. **Create `.env` File**:
   ```bash
   echo "HF_TOKEN=your_token_here" > .env
   ```

## 🎯 Usage

### Basic Usage
```bash
python meeting_transcriber.py
```

### Whisper Model Selection
Edit the `main()` function in `meeting_transcriber.py` to choose different models:

```python
# RECOMMENDED (English-only, good accuracy + speed):
transcriber = MeetingTranscriber(model_name="openai/whisper-medium.en", debug=True)

# HIGHEST ACCURACY:
transcriber = MeetingTranscriber(model_name="openai/whisper-large-v3")

# FASTEST:
transcriber = MeetingTranscriber(model_name="openai/whisper-base")
```

**Model Comparison:**
| Model | Accuracy | Speed | VRAM | Best For |
|-------|----------|-------|------|----------|
| `openai/whisper-large-v3` | Highest | Slowest | ~10GB | Maximum accuracy |
| `openai/whisper-medium.en` | Good | Fast | ~5GB | **Recommended for English** |
| `openai/whisper-base` | Basic | Very fast | ~1GB | Quick transcription |

### Debug Mode
Enable debug mode to see detailed processing information:

```python
# Enable debug mode in meeting_transcriber.py
transcriber = MeetingTranscriber(debug=True)
```

**Debug mode shows:**
- 🎯 Diarization processing details
- 📊 Speaker segment counts and timing
- 🎤 ASR chunk processing
- 🔍 Speaker-to-text alignment details
- ✅ Speaker consistency tracking

**Normal mode shows:**
- Real-time transcription with speaker labels and timestamps
- Final conversation summary
- Essential system notifications

### During Recording
- **Automatic Detection**: System automatically detects and labels speakers as "Person 1", "Person 2", etc.
- **No Setup Required**: No need for introductions or voice training
- **Consistent Labels**: Same person gets same label throughout the conversation
- **Real-time Output**: See transcription appear as people speak

### Stop & Get Summary
- Press `Ctrl+C` to stop recording
- Automatic final summary of the entire conversation
- Transcription saved to timestamped file

## 🎯 Speaker Identification

### How It Works
The integrated pipeline uses **timestamp-based alignment** instead of complex voice analysis:

1. **Diarization First**: pyannote identifies "when" each speaker talks
2. **ASR Second**: Whisper transcribes "what" was said with timestamps  
3. **Smart Alignment**: System matches text to speakers based on time overlap
4. **Consistent Mapping**: Same speaker gets same "Person X" label across all chunks

### Speaker Labels
- **Person 1**: First speaker detected
- **Person 2**: Second speaker detected
- **Person 3+**: Additional speakers as they appear
- **Consistent**: Same person keeps same label throughout entire conversation

### Advantages Over Voice Analysis
- **More Reliable**: No complex voice similarity calculations that can fail
- **Faster Processing**: Timestamp alignment is much faster than voice comparison
- **Better Accuracy**: Uses state-of-the-art diarization models trained specifically for speaker detection
- **Fewer Errors**: Eliminates voice comparison edge cases and similarity threshold issues

## 🔧 Troubleshooting

### Model Access Issues
```
Error: Hugging Face token not found
```
**Solution**: Ensure `.env` file exists with valid `HF_TOKEN=your_token`

### Permission Errors
```
Cannot access model pyannote/speaker-diarization-3.1
```
**Solution**: Accept user agreement for the diarization model on Hugging Face

### Audio Issues
- **No microphone detected**: Check system audio permissions
- **Poor speaker detection**: Ensure clear audio with minimal background noise
- **Processing delays**: 10-15 second delay is normal for real-time processing

### Performance Notes
- First run downloads ~3GB of AI models (one-time setup)
- Requires ~6GB RAM for optimal performance
- GPU acceleration automatically used if available

### Installation Issues
```
ERROR: Cannot import 'setuptools.build_meta'
```
**Solution**: Upgrade build tools first:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

```
Python 3.13 compatibility issues
```
**Solution**: Use Python 3.8-3.11:
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📋 Technical Details

### Pipeline Architecture
Based on the [Hugging Face ASR + Diarization blog post](https://huggingface.co/blog/asr-diarization), this implementation uses:

- **ASR Pipeline**: Whisper models via transformers pipeline
- **Diarization Pipeline**: pyannote/speaker-diarization-3.1
- **Alignment Algorithm**: Time-based overlap matching
- **Consistency Tracking**: Chunk-based speaker mapping

### Models Used
- **Transcription**: OpenAI Whisper (configurable model)
- **Speaker Diarization**: pyannote/speaker-diarization-3.1
- **Summarization**: knkarthick/meeting-summary-samsum

### Audio Processing
- **Sample Rate**: 16kHz
- **Chunk Size**: 10 seconds
- **Overlap Threshold**: 10% minimum for speaker-text alignment
- **Processing**: Real-time with ~10-15 second delay

### System Requirements
- **Python**: 3.8-3.11
- **RAM**: 6GB+ recommended
- **Storage**: 4GB for models
- **OS**: Windows, macOS, Linux
- **GPU**: Optional (CUDA acceleration if available)

## 🆚 Why This Approach?

### Previous Challenges
The original implementation used complex voice characteristic analysis which had issues:
- Voice similarity calculations could be unreliable
- Complex threshold tuning required
- Edge cases with similar voices or voice changes
- Performance overhead from voice comparison algorithms

### New Integrated Approach
The current implementation uses proven diarization models:
- **More Accurate**: Uses models trained specifically for speaker identification
- **More Reliable**: Timestamp-based alignment eliminates voice comparison errors  
- **Faster**: No complex similarity calculations
- **Simpler**: Clean architecture with fewer failure points
- **Industry Standard**: Same approach used by professional transcription services

## 📝 Output Files

Generated files are saved with timestamps:
- **Format**: `meeting-transcription-MM-DD-YY-HHMM.txt`
- **Contents**: 
  - Meeting summary at the top
  - Full transcription with speaker labels and timestamps
  - Processing metadata

## 📝 License

This project is provided as-is for educational and research purposes.

## 🙋 Support

If you encounter issues:
1. Check that Hugging Face model agreement is accepted
2. Verify your `.env` file contains a valid token
3. Ensure microphone permissions are granted
4. Try using debug mode to see detailed processing information

---

*Built with ❤️ using the Hugging Face ASR + Diarization pipeline approach.*