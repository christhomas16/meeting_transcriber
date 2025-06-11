# Real-Time Meeting Transcriber with Smart Speaker Identification

A sophisticated real-time meeting transcriber that captures microphone audio, transcribes speech using Whisper, and intelligently identifies speakers by name and voice characteristics.

## ✨ Features

- **🎙️ Real-Time Transcription**: Live speech-to-text using OpenAI Whisper
- **👥 Unlimited Speaker Support**: Configurable speaker detection (default: 15 speakers, customizable up to 50+)
- **🧠 Smart Speaker Recognition**: Identifies speakers by self-introductions ("My name is John") and voice characteristics
- **🔍 Voice Memory**: Maintains consistent speaker identity throughout the conversation
- **📝 Clean Conversation Output**: Natural speech segments without artificial breaks
- **📊 Final Summary**: Comprehensive conversation summary when you end the session
- **🔄 Continuous Processing**: 10-second audio chunks for optimal balance of accuracy and real-time performance

## 🚀 How It Works

The system combines multiple AI models to create intelligent speaker-aware transcription:

1. **Audio Capture**: Records from your microphone in 10-second chunks
2. **Speaker Diarization**: Uses pyannote.audio to detect when different people are speaking
3. **Speech Transcription**: Converts audio to text using Whisper base model
4. **Smart Speaker Matching**: 
   - Detects self-introductions like "My name is John" or "I'm Anna"
   - Analyzes voice characteristics (pitch, frequency distribution, energy patterns)
   - Maintains consistent speaker identification across audio chunks
5. **Natural Segmentation**: Preserves Whisper's natural speech boundaries
6. **Final Summarization**: Generates conversation summary using specialized meeting summary model

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Microphone access
- Internet connection (for model downloads)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd meeting-transcriber
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Hugging Face Configuration
**⚠️ Critical Step**: Speaker diarization requires Hugging Face access.

1. **Create Account & Token**:
   - Sign up at [huggingface.co](https://huggingface.co)
   - Generate token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

2. **Accept Model Terms** (Required for all three):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/embedding](https://huggingface.co/pyannote/embedding)

3. **Create `.env` File**:
   ```bash
   echo "HF_TOKEN=your_token_here" > .env
   ```

## 🎯 Usage

### Basic Usage
```bash
python meeting_transcriber.py
```
*Default: Supports up to 15 speakers*

### Custom Speaker Limits
For different meeting sizes, edit the `main()` function in `meeting_transcriber.py`:

```python
# Small meetings (2-5 people) - Better accuracy
transcriber = MeetingTranscriber(max_speakers_limit=5)

# Medium groups (6-10 people) - Good balance  
transcriber = MeetingTranscriber(max_speakers_limit=10)

# Large conferences (15+ people) - Maximum coverage
transcriber = MeetingTranscriber(max_speakers_limit=25)

# No practical limit (50+ people) - May impact performance
transcriber = MeetingTranscriber(max_speakers_limit=50)
```

### During Recording
- **No preparation needed** - System automatically detects and labels speakers
- **Optional introductions** - Say "My name is [Name]" for personalized labels
- **Anonymous speakers** - Automatically labeled as "Person 1", "Person 2", etc.
- **Late introductions** - Names can be added anytime during the conversation

### Stop & Get Summary
- Press `Ctrl+C` to stop
- Automatic final summary of the entire conversation

## 💡 Smart Speaker Features

### Name Recognition
The system automatically detects introductions:
- "My name is John" → Labeled as "John"
- "I'm Anna" → Labeled as "Anna"  
- "This is Mike" → Labeled as "Mike"

### Anonymous Speaker Handling
**What happens when people DON'T introduce themselves?**
- **First speaker**: Automatically labeled "Person 1"
- **Second speaker**: Automatically labeled "Person 2"  
- **Additional speakers**: Continue as "Person 3", "Person 4", etc.
- **Voice consistency**: Uses voice characteristics to maintain the same label throughout
- **Late introductions**: If someone says their name later, their label updates to their actual name

### Voice Consistency  
- Analyzes voice characteristics (pitch, frequency, energy patterns)
- Maintains speaker identity even with inconsistent diarization labels
- Works reliably whether speakers introduce themselves or remain anonymous

### Conversation Flow
- Preserves natural speech patterns
- No artificial sentence breaking
- All transcribed text captured accurately

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
**Solution**: Accept user agreements for all three required models on Hugging Face

### Audio Issues
- **No microphone detected**: Check system audio permissions
- **Poor speaker detection**: Ensure clear audio with minimal background noise
- **Speaker confusion**: Have speakers introduce themselves by name

### Performance Notes
- First run downloads ~2GB of AI models (one-time setup)
- Requires ~4GB RAM for optimal performance
- Processing delay of 10-15 seconds is normal for real-time transcription

## 📋 Technical Details

### Models Used
- **Transcription**: OpenAI Whisper (base model)
- **Speaker Diarization**: pyannote/speaker-diarization-3.1
- **Voice Embedding**: pyannote/embedding  
- **Summarization**: knkarthick/meeting-summary-samsum

### Audio Processing
- **Sample Rate**: 16kHz
- **Chunk Size**: 10 seconds
- **Maximum Speakers**: Configurable (default: 15, supports 50+)
- **Minimum Speaker Segment**: 0.3 seconds
- **Voice Similarity Threshold**: 40%
- **Adaptive Detection**: Dynamically adjusts based on detected speakers

### System Requirements
- **Python**: 3.8+
- **RAM**: 4GB+ recommended
- **Storage**: 3GB for models
- **OS**: Windows, macOS, Linux


## 📝 License

This project is provided as-is for educational and research purposes.

## 🙋 Support

If you encounter issues:
1. Check that all Hugging Face model agreements are accepted
2. Verify your `.env` file contains a valid token
3. Ensure microphone permissions are granted
4. Try restarting if models seem corrupted

---

*Built with ❤️ using Whisper, pyannote.audio, and advanced voice processing techniques.*