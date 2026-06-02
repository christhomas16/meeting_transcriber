# Real-Time Meeting Transcriber with Advanced Speaker Identification

A sophisticated real-time meeting transcriber that provides both a live preview of the conversation and a final, highly accurate transcript with speaker identification powered by voice embeddings.

## ⚡ Two engines

| Engine | ASR | Diarization | Alignment | When to use |
|--------|-----|-------------|-----------|-------------|
| **Modern** (`--modern`, 2026) | NVIDIA **Parakeet-TDT-0.6b-v3** via `parakeet-mlx` (native Apple-GPU) | **pyannote/speaker-diarization-community-1** | Sentence-level (WhisperX-style) | **Recommended on Apple Silicon.** Faster and more accurate. |
| **Legacy** (default) | `openai/whisper-medium.en` (transformers) | `pyannote/speaker-diarization-3.1` + voiceprints | Chunk-level overlap | Fallback / non-MLX hosts. |

The two engines use **separate virtual environments** (`venv-modern` vs `venv`) so their dependencies never conflict. `run.sh --modern` creates and uses `venv-modern` automatically. See [Modern engine setup](#-modern-engine-2026-setup).

## ✨ Features

- **🎙️ Live Transcription Preview** (Optional): Get a real-time feed of the conversation as it happens.
- **🎯 Advanced Speaker Identification**: Uses voice embeddings to accurately distinguish between multiple speakers.
- **📝 High-Quality Local Summarization**: Leverages a local LLM via **Ollama** (e.g., `llama3.2`, `gemma2:2b`) for a superior meeting summary.
- **🗣️ High-Quality Final Transcript**: Generates an accurate, speaker-separated transcript after the meeting concludes.
- **📝 Automatic Meeting Summary**: A summary of the key points and speakers is automatically generated at the end.
- **📋 Session Summary of Summaries**: Creates a comprehensive overview of all meetings from a session using AI.
- **💾 Automatic File Organization**: The complete summary and transcript are saved to separate files in a `transcriptions/` folder.
- **⚙️ Simple & Robust**: Records for up to 5 minutes or until `Ctrl+C` is pressed, then processes the entire conversation at once.
- **🛡️ Graceful Shutdown**: Always processes current recording before shutting down, ensuring no audio is lost.

## 🚀 How It Works

The system runs two parallel processes:

1. **Live Preview Thread** (Optional):
   * Audio is captured in small, overlapping 5-second chunks.
   * A simple energy-based Voice Activity Detection (VAD) ignores silent chunks.
   * Non-silent chunks are fed to a Whisper ASR model for a fast, real-time preview.
   * This output is for **display only** and is de-duplicated for clean viewing.
   * **Disabled by default** for better performance.

2. **Final Processing**:
   * The entire conversation is recorded into a single, high-quality audio buffer.
   * When the recording stops (via `Ctrl+C` or the 5-minute timer), this full buffer is processed.
   * **Diarization**: `pyannote/speaker-diarization-3.1` identifies *who* spoke and *when*.
   * **Voiceprinting**: A speaker embedding model (`pyannote/embedding`) creates a unique voiceprint for each speaker segment.
   * **Intelligent Matching**: A tiered confidence system compares these voiceprints to robustly identify speakers, even distinguishing between multiple people with similar voices.
   * **ASR Transcription**: The Whisper model transcribes the full audio with long-form audio support for high accuracy.
   * **Final Alignment**: The diarization, voiceprints, and transcription are combined to produce the final, accurate transcript.
   * **Summarization**: The final transcript is sent to a locally running Large Language Model (LLM) via **Ollama** to generate a high-quality, context-aware summary.
   * **Session Summary**: When the application shuts down, it creates a comprehensive AI-generated summary of all meetings from the session.

## 🛠️ Installation

### Prerequisites
- Python 3.8-3.11
- A working microphone
- An internet connection for model downloads on first run.
- **Ollama**: For local LLM summarization. See step 5.

### 1. Clone the Repository
```bash
git clone https://github.com/christhomas16/meeting_transcriber.git
cd meeting_transcriber
```

### 2. Quick Start with run.sh (Recommended)
The easiest way to get started is using the provided `run.sh` script:

```bash
# ⭐ Passive meeting capture (recommended):
#    records the whole meeting, press Ctrl+C when done, then you get one
#    transcript + one summary. Uses the modern engine + venv-modern.
./run.sh --meeting

# Show help and all options
./run.sh --help

# Standard mode (5-minute chunks), legacy engine
./run.sh

# Enable live transcription (legacy engine only)
./run.sh --live --debug

# Use a different summary model
./run.sh --meeting --ollama-model qwen3:4b
```

> **Meeting mode** is the intended day-to-day workflow: start it before the call,
> let it listen passively, and press **Ctrl+C** when the meeting ends. It records
> one continuous session (no 5-minute chunking, no audio gaps) and then produces a
> single transcript and summary. The first run creates `venv-modern` and downloads
> models. For per-speaker labels, accept the
> [community-1 terms](https://huggingface.co/pyannote/speaker-diarization-community-1)
> on Hugging Face — otherwise the transcript is single-speaker.

### 3. Manual Setup (Alternative)
If you prefer manual setup:

```bash
# Create a virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate
# On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Hugging Face Access
**⚠️ This is a critical step.** The diarization and embedding models require authentication with Hugging Face.

1. **Create a Hugging Face Account**: If you don't have one, sign up at [huggingface.co](https://huggingface.co).
2. **Generate an Access Token**: Go to your settings and create a new access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. **Accept Model User Agreements**: You must visit and accept the terms of service for the following two models:
   * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   * [pyannote/embedding](https://huggingface.co/pyannote/embedding)
4. **Create a `.env` File**: In the root of the project, create a file named `.env` and add your token to it:
   ```
   HF_TOKEN=your_token_here
   ```

### 5. Install and Set Up Ollama (for Summarization)
For the best summary quality, this tool uses a local LLM served by [Ollama](https://ollama.com).

1. **Install Ollama**: Follow the installation instructions on their website.
2. **Pull a Model**: You need at least one model for summarization. The script tries your specified model first, then falls back through a chain of modern and lightweight models (`qwen3:4b` → `gemma3:4b` → `llama3.2` → `gemma2:2b`), skipping any you don't have installed. Long transcripts are summarized via **map-reduce** (chunk summaries → final summary) so they never overflow a small model's context window.
   ```bash
   # Recommended modern models (2026)
   ollama pull qwen3:4b
   ollama pull gemma3:4b

   # Lighter-weight / fallback models
   ollama pull llama3.2
   ollama pull gemma2:2b
   ```
3. **Ensure Ollama is Running**: Before running the transcriber, make sure the Ollama application is running in the background.

## 🚀 Modern engine (2026) setup

The modern engine runs in its own environment so it never clashes with the legacy pins.

```bash
# One-time: accept the gated model terms on Hugging Face (your HF_TOKEN must have access).
#   https://huggingface.co/pyannote/speaker-diarization-community-1
# Note: with pyannote.audio 4.0 the 3.1 pipeline also pulls shared community-1 assets,
# so accepting community-1 is required for any diarization.

# Easiest — run.sh sets up venv-modern for you:
./run.sh --modern

# Manual setup:
python3.11 -m venv venv-modern
source venv-modern/bin/activate
pip install -r requirements-modern.txt
python meeting_transcriber.py --modern
```

You can also run the engine standalone on an existing audio file (great for validating quality):

```bash
source venv-modern/bin/activate
python modern_transcribe.py path/to/meeting.wav --debug
```

If the diarization model can't be loaded (terms not accepted / offline), the modern engine
**falls back to ASR-only** (single speaker) rather than failing the run.

## 🎯 Usage

### Running the Transcriber

#### Using run.sh (Recommended)
```bash
# Show all available options
./run.sh --help

# Default usage (no live transcription)
./run.sh

# With live transcription
./run.sh --live

# With live transcription and debug mode
./run.sh --live --debug

# Using different models
./run.sh --model openai/whisper-large-v3 --ollama-model gemma2:2b
```

#### Direct Python Usage
```bash
# Default: no live transcription
python meeting_transcriber.py

# Enable live transcription
python meeting_transcriber.py --live

# Enable live transcription with debug mode
python meeting_transcriber.py --live --debug

# Use different models
python meeting_transcriber.py --model openai/whisper-large-v3 --ollama-model gemma2:2b
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--live`, `--live-transcription` | Enable live transcription preview | Disabled |
| `--debug` | Enable debug mode for detailed processing output | Disabled |
| `--model MODEL` | Specify Whisper model to use | `openai/whisper-medium.en` |
| `--ollama-model MODEL` | Specify Ollama model for summarization | `llama3.2` |
| `--help`, `-h` | Show help message | - |

### Whisper Model Options

| Model | Accuracy | Speed | VRAM | Notes |
|-------|----------|-------|------|-------|
| `openai/whisper-large-v3` | Highest | Slowest | ~10GB | Best for offline, max-quality transcription. |
| `openai/whisper-medium.en` | Excellent | Fast | ~5GB | **Recommended for English meetings.** |
| `openai/whisper-base` | Good | Fastest | ~1GB | Good for development or less powerful machines. |

### Ollama Model Options

| Model | Quality | Speed | RAM | Notes |
|-------|---------|-------|-----|-------|
| `llama3.2` | Excellent | Fast | ~8GB | **Recommended default.** |
| `gemma2:2b` | Good | Fastest | ~4GB | Good for lighter-weight systems. |
| `magistral` | Highest | Slowest | ~16GB | Best quality, requires more resources. |

### What Happens During Recording

1. **Recording Starts**: The app begins recording audio from your microphone
2. **Live Preview** (if enabled): Shows real-time transcription with `[Live]:` prefix
3. **Recording Stops**: Either after 5 minutes or when you press `Ctrl+C`
4. **Processing**: Background processing of the full audio with ASR + Diarization
5. **Results**: Final transcript and summary are displayed and saved to files

### File Output

The app creates organized files in the `transcriptions/` folder:

- **`meeting-summary-{timestamp}.txt`**: AI-generated summary of the meeting
- **`meeting-transcription-{timestamp}.txt`**: Full transcript with speaker identification
- **`full-meeting-summary-{timestamp}.txt`**: Session summary of all meetings (created on shutdown)

### Session Summary

When you exit the application (Ctrl+C), it automatically:
1. Processes any current recording
2. Creates individual meeting summaries
3. Generates an AI-powered session summary of all meetings
4. Saves everything to organized files

## 🔧 Troubleshooting

- **"Ollama server not running"**: Make sure you have started the Ollama application before running the script.
- **"Model not found" Error during Summary**: This means the model specified in the script (e.g., `llama3.2`) is not present in your local Ollama instance. Use `ollama pull <model_name>` to download it, or edit the script to use a model you have.
- **Dependency Conflicts on Install**: If `pip install` fails, ensure your `torch` and `torchaudio` versions are compatible. The included `requirements.txt` should handle this.
- **Model Access Errors**: If you see errors related to `401` or permissions, double-check that you have accepted the user agreements for both `pyannote` models and that your `HF_TOKEN` in the `.env` file is correct.
- **Missing Words in Live Preview**: The Voice Activity Detection (VAD) threshold might be too aggressive for your microphone. You can make it more sensitive by lowering the `vad_threshold` value in the `__init__` method.
- **Poor Speaker Identification**: Ensure your microphone provides clear audio with minimal background noise. The embedding models work best when they can get a clean sample of each person's voice.
- **Short Audio Segment Errors**: The app automatically handles very short audio segments that might cause embedding errors.

## 📝 License

This project is open-source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📊 Recent Updates

- ✅ **Live transcription disabled by default** for better performance
- ✅ **Command-line options** for flexible configuration
- ✅ **Session summary of summaries** with AI-powered analysis
- ✅ **Improved run.sh script** with help system and validation
- ✅ **Graceful shutdown** that always processes current recording
- ✅ **Organized file output** in transcriptions folder
- ✅ **Enhanced error handling** for short audio segments
