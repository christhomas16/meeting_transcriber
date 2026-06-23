#!/bin/bash

# Default values
LIVE_TRANSCRIPTION=false
DEBUG_MODE=false
WHISPER_MODEL="openai/whisper-medium.en"
OLLAMA_MODEL="llama3.2"
SHOW_HELP=false
MODERN=false
MEETING=false
DEVICE=""

# Function to display help
show_help() {
    echo "🎙️  Meeting Transcriber - Setup and Run Script"
    echo ""
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --meeting                       Passive meeting capture: record the whole meeting,"
    echo "                                  press Ctrl+C to stop, then get one transcript +"
    echo "                                  summary. Uses the modern engine. [RECOMMENDED]"
    echo "  --modern                        Use the 2026 engine: Parakeet-MLX ASR + pyannote"
    echo "                                  community-1 diarization (uses venv-modern)"
    echo "  --live, --live-transcription    Enable live transcription preview (legacy engine only)"
    echo "  --debug                         Enable debug mode for detailed processing output"
    echo "  --model MODEL                   Specify Whisper model (legacy engine, default: openai/whisper-medium.en)"
    echo "  --ollama-model MODEL            Specify Ollama model (default: llama3.2)"
    echo "  --help, -h                      Show this help message"
    echo ""
    echo "Whisper Model Options:"
    echo "  openai/whisper-large-v3         Best accuracy, latest model (~10GB VRAM)"
    echo "  openai/whisper-medium.en        Excellent accuracy, fast (~5GB VRAM) [DEFAULT]"
    echo "  openai/whisper-base             Good accuracy, fastest (~1GB VRAM)"
    echo ""
    echo "Ollama Model Options:"
    echo "  llama3.2                        Good balance of quality and speed [DEFAULT]"
    echo "  gemma2:2b                       Lightweight, good for basic summaries"
    echo "  magistral                       High quality, requires >16GB RAM"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                        # Default: no live transcription"
    echo "  ./run.sh --meeting              # Passive meeting capture (recommended)"
    echo "  ./run.sh --modern               # Use the 2026 Parakeet + community-1 engine"
    echo "  ./run.sh --live                 # Enable live transcription (legacy engine)"
    echo "  ./run.sh --live --debug         # Enable both live transcription and debug mode"
    echo "  ./run.sh --model openai/whisper-large-v3  # Use larger model"
    echo "  ./run.sh --ollama-model gemma2:2b         # Use different Ollama model"
    echo ""
    echo "Prerequisites:"
    echo "  - Python 3.8-3.11"
    echo "  - Working microphone"
    echo "  - Hugging Face token in .env file (accept pyannote model terms on HF)"
    echo "  - Ollama running with at least one model"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --live|--live-transcription)
            LIVE_TRANSCRIPTION=true
            shift
            ;;
        --modern)
            MODERN=true
            shift
            ;;
        --meeting)
            MEETING=true
            MODERN=true   # meeting mode uses the modern engine + venv-modern
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --list-devices)
            DEVICE="__list__"
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --model)
            WHISPER_MODEL="$2"
            shift 2
            ;;
        --ollama-model)
            OLLAMA_MODEL="$2"
            shift 2
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$SHOW_HELP" = true ]; then
    show_help
    exit 0
fi

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 is not installed or not in PATH"
    echo "Please install Python 3.11 and try again"
    exit 1
fi

# Select the environment based on engine: the modern engine uses its own venv so
# it never conflicts with the legacy dependency pins.
if [ "$MODERN" = true ]; then
    VENV_DIR="venv-modern"
    REQ_FILE="requirements-modern.txt"
else
    VENV_DIR="venv"
    REQ_FILE="requirements.txt"
fi

# Check if virtual environment already exists
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment ($VENV_DIR)..."
    python3.11 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    NEW_VENV=true
else
    echo "📦 Virtual environment already exists ($VENV_DIR)"
    NEW_VENV=false
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Install dependencies on first creation
if [ "$NEW_VENV" = true ]; then
    echo "📦 Installing dependencies from $REQ_FILE..."
    pip install -r "$REQ_FILE"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
else
    echo "📦 Dependencies already installed"
fi

# Build command line arguments for Python script.
# Use an array so values containing spaces (e.g. --device "Aggregate Device")
# are passed as a single argument rather than being word-split.
PYTHON_ARGS=()
if [ "$LIVE_TRANSCRIPTION" = true ]; then
    PYTHON_ARGS+=(--live)
fi

if [ "$MODERN" = true ]; then
    PYTHON_ARGS+=(--modern)
fi

if [ "$MEETING" = true ]; then
    PYTHON_ARGS+=(--meeting)
fi

if [ "$DEVICE" = "__list__" ]; then
    PYTHON_ARGS+=(--list-devices)
elif [ -n "$DEVICE" ]; then
    PYTHON_ARGS+=(--device "$DEVICE")
fi

if [ "$DEBUG_MODE" = true ]; then
    PYTHON_ARGS+=(--debug)
fi

if [ "$WHISPER_MODEL" != "openai/whisper-medium.en" ]; then
    PYTHON_ARGS+=(--model "$WHISPER_MODEL")
fi

if [ "$OLLAMA_MODEL" != "llama3.2" ]; then
    PYTHON_ARGS+=(--ollama-model "$OLLAMA_MODEL")
fi

# Display configuration
echo ""
echo "🚀 Starting Meeting Transcriber with configuration:"
echo "   🎧 Mode: $([ "$MEETING" = true ] && echo "MEETING (passive capture, one transcript + summary)" || echo "STANDARD (5-minute chunks)")"
echo "   ⚙️  Engine: $([ "$MODERN" = true ] && echo "MODERN (Parakeet-MLX + community-1)" || echo "LEGACY (Whisper + pyannote 3.1)")"
echo "   📺 Live transcription: $([ "$LIVE_TRANSCRIPTION" = true ] && echo "ENABLED" || echo "DISABLED")"
echo "   🐛 Debug mode: $([ "$DEBUG_MODE" = true ] && echo "ENABLED" || echo "DISABLED")"
echo "   🤖 Whisper model: $WHISPER_MODEL"
echo "   🧠 Ollama model: $OLLAMA_MODEL"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Make sure you have created a .env file with your HF_TOKEN"
    echo "   Example: echo 'HF_TOKEN=your_token_here' > .env"
    echo ""
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434 > /dev/null; then
    echo "⚠️  Warning: Ollama server not running"
    echo "   Start Ollama to enable AI-powered summaries"
    echo "   The app will still work for transcription without summaries"
    echo ""
fi

# Run the meeting transcriber
echo "🎙️  Starting Meeting Transcriber..."
python meeting_transcriber.py "${PYTHON_ARGS[@]}"