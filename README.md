# Real-Time Meeting Transcriber with Advanced Speaker Identification

A sophisticated real-time meeting transcriber that provides both a live preview of the conversation and a final, highly accurate transcript with speaker identification powered by voice embeddings.

## ✨ Features

- **🎙️ Live Transcription Preview**: Get a real-time feed of the conversation as it happens.
- **🎯 Advanced Speaker Identification**: Uses voice embeddings to accurately distinguish between multiple speakers.
- **📝 High-Quality Local Summarization**: Leverages a local LLM via **Ollama** (e.g., `llama3.2`, `gemma2:2b`) for a superior meeting summary.
- **🗣️ High-Quality Final Transcript**: Generates an accurate, speaker-separated transcript after the meeting concludes.
- **📝 Automatic Meeting Summary**: A summary of the key points and speakers is automatically generated at the end.
- **💾 Automatic File Saving**: The complete summary and transcript are saved to a timestamped text file.
- **⚙️ Simple & Robust**: Records for up to 10 minutes or until `Ctrl+C` is pressed, then processes the entire conversation at once.

## 🚀 How It Works

The system now runs two parallel processes:

1.  **Live Preview Thread**:
    *   Audio is captured in small, overlapping 5-second chunks.
    *   A simple energy-based Voice Activity Detection (VAD) ignores silent chunks.
    *   Non-silent chunks are fed to a Whisper ASR model for a fast, real-time preview.
    *   This output is for **display only** and is de-duplicated for clean viewing.

2.  **Final Processing**:
    *   The entire conversation is recorded into a single, high-quality audio buffer.
    *   When the recording stops (via `Ctrl+C` or the 10-minute timer), this full buffer is processed.
    *   **Diarization**: `pyannote/speaker-diarization-3.1` identifies *who* spoke and *when*.
    *   **Voiceprinting**: A speaker embedding model (`pyannote/embedding`) creates a unique voiceprint for each speaker segment.
    *   **Intelligent Matching**: A tiered confidence system compares these voiceprints to robustly identify speakers, even distinguishing between multiple people with similar voices.
    *   **ASR Transcription**: The Whisper model transcribes the full audio with long-form audio support for high accuracy.
    *   **Final Alignment**: The diarization, voiceprints, and transcription are combined to produce the final, accurate transcript.
    *   **Summarization**: The final transcript is sent to a locally running Large Language Model (LLM) via **Ollama** to generate a high-quality, context-aware summary.

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

### 2. Set Up Virtual Environment
```bash
# Create a virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate
# On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Hugging Face Access
**⚠️ This is a critical step.** The diarization and embedding models require authentication with Hugging Face.

1.  **Create a Hugging Face Account**: If you don't have one, sign up at [huggingface.co](https://huggingface.co).
2.  **Generate an Access Token**: Go to your settings and create a new access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3.  **Accept Model User Agreements**: You must visit and accept the terms of service for the following two models:
    *   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    *   [pyannote/embedding](https://huggingface.co/pyannote/embedding)
4.  **Create a `.env` File**: In the root of the project, create a file named `.env` and add your token to it:
    ```
    HF_TOKEN=your_token_here
    ```

### 5. Install and Set Up Ollama (for Summarization)
For the best summary quality, this tool uses a local LLM served by [Ollama](https://ollama.com).

1.  **Install Ollama**: Follow the installation instructions on their website.
2.  **Pull a Model**: You need at least one model for summarization. We recommend `llama3.2` as a great starting point. The script will try your specified model first, then fall back to `gemma2:2b` if the first one fails.
    ```bash
    # Recommended default model
    ollama pull llama3.2

    # Fallback model for lighter-weight systems
    ollama pull gemma2:2b

    # Optional, more powerful model (requires >16GB RAM)
    ollama pull magistral
    ```
3.  **Ensure Ollama is Running**: Before running the transcriber, make sure the Ollama application is running in the background.

## 🎯 Usage

### Running the Transcriber
Make sure your Ollama application is running, then simply run the script from your terminal:
```bash
python meeting_transcriber.py
```
- A `[Live]:` feed will show the real-time transcription.
- When you're done, press `Ctrl+C` to stop the recording.
- The final, high-quality diarized transcript will then be processed and displayed.
- The LLM-generated summary will be streamed to your console.
- The full summary and transcript will be saved to a `meeting-transcription-*.txt` file.

### Selecting Models
You can easily change the Whisper and Ollama models used by editing the `main()` function in `meeting_transcriber.py`:

```python
# In main():
transcriber = MeetingTranscriber(
    model_name="openai/whisper-medium.en",
    debug=False,
    ollama_model="llama3.2" # Or "gemma2:2b", "magistral", etc.
)
```

| Model | Accuracy | Speed | VRAM | Notes |
|---|---|---|---|---|
| `openai/whisper-large-v3` | Highest | Slowest | ~10GB | Best for offline, max-quality transcription. |
| `openai/whisper-medium.en`| Excellent| Fast | ~5GB | **Recommended for English meetings.** |
| `openai/whisper-base` | Good | Fastest | ~1GB | Good for development or less powerful machines. |

### Enabling Debug Mode
To see detailed processing information, including voiceprint similarity scores, set `debug=True` when creating the `transcriber` instance.

## 🔧 Troubleshooting

- **"Ollama server not running"**: Make sure you have started the Ollama application before running the script.
- **"Model not found" Error during Summary**: This means the model specified in the script (e.g., `llama3.2`) is not present in your local Ollama instance. Use `ollama pull <model_name>` to download it, or edit the script to use a model you have.
- **Dependency Conflicts on Install**: If `pip install` fails, ensure your `torch` and `torchaudio` versions are compatible. The included `requirements.txt` should handle this.
- **Model Access Errors**: If you see errors related to `401` or permissions, double-check that you have accepted the user agreements for both `pyannote` models and that your `HF_TOKEN` in the `.env` file is correct.
- **Missing Words in Live Preview**: The Voice Activity Detection (VAD) threshold might be too aggressive for your microphone. You can make it more sensitive by lowering the `vad_threshold` value in the `__init__` method.
- **Poor Speaker Identification**: Ensure your microphone provides clear audio with minimal background noise. The embedding models work best when they can get a clean sample of each person's voice.

## 📝 License

This project is open-source and available under the MIT License.
