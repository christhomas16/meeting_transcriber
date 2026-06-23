#!/usr/bin/env python3
"""
Modern transcription engine (2026 stack)
========================================

Next-generation ASR + diarization engine for the meeting transcriber, usable two
ways:

  1. Standalone CLI to validate quality on an audio file:
         python modern_transcribe.py path/to/meeting.wav --debug

  2. Imported by meeting_transcriber.py (run with --modern) so the live mic loop,
     summaries, and file output all use it. The engine exposes an in-memory
     transcribe_array() so no temp WAV is needed for live capture.

What changed vs. the original whisper-medium.en + pyannote-3.1 pipeline
-----------------------------------------------------------------------
- ASR:        NVIDIA Parakeet-TDT-0.6b-v3 via parakeet-mlx (native Apple GPU,
              SOTA English WER, native word/sentence timestamps).
- Diarization: pyannote/speaker-diarization-community-1 (pyannote.audio 4.0).
- Alignment:  WhisperX-style. Each ASR *sentence* (clean text + timestamps) is
              attributed to the speaker whose diarization segment contains the
              sentence midpoint, then consecutive same-speaker sentences are
              merged into turns. Far more accurate at speaker boundaries than the
              legacy "one whole 5-minute chunk -> one speaker" assignment.

Install (separate from the legacy requirements.txt):
    python3.11 -m venv venv-modern
    source venv-modern/bin/activate
    pip install -r requirements-modern.txt

Requires HF_TOKEN in .env and acceptance of the community-1 model terms at
https://huggingface.co/pyannote/speaker-diarization-community-1
"""

import argparse
import os
import sys

import numpy as np

DEFAULT_ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
DEFAULT_DIAR_MODEL = "pyannote/speaker-diarization-community-1"


def pick_device():
    """Return the best available torch device string."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ModernEngine:
    """Loads the ASR and diarization models once and reuses them for every call."""

    def __init__(self, asr_model=DEFAULT_ASR_MODEL, diar_model=DEFAULT_DIAR_MODEL,
                 hf_token=None, device="cpu", debug=False):
        self.debug = debug
        self.device = device

        from parakeet_mlx import from_pretrained
        if self.debug:
            print(f"🎤 Loading ASR model: {asr_model}")
        self.asr_model = from_pretrained(asr_model)
        self.asr_sample_rate = self.asr_model.preprocessor_config.sample_rate

        # Long recordings (meetings) are transcribed with parakeet's built-in
        # overlapping-window chunking so they don't process as one giant mel.
        self.long_audio_threshold_s = 480   # 8 minutes
        self.asr_chunk_duration = 120.0     # 2-minute windows
        self.asr_overlap_duration = 15.0

        # Diarization is optional: if the gated model can't be loaded (terms not
        # accepted, offline, etc.) we still return an ASR-only transcript rather
        # than failing the whole run.
        self.diar_pipeline = None
        try:
            import torch
            from pyannote.audio import Pipeline
            if self.debug:
                print(f"🎯 Loading diarization model: {diar_model}")
            self.diar_pipeline = Pipeline.from_pretrained(diar_model, token=hf_token)
            if device in ("mps", "cuda"):
                self.diar_pipeline.to(torch.device(device))
        except Exception as e:
            print(f"⚠️  Diarization unavailable ({e}). Falling back to ASR-only "
                  f"(single speaker).", file=sys.stderr)

    # --- ASR -------------------------------------------------------------

    def _asr_sentences_from_file(self, path):
        # chunk_duration only kicks in for files longer than it; short files are
        # still processed whole, so this is always safe.
        result = self.asr_model.transcribe(
            path, chunk_duration=self.asr_chunk_duration, overlap_duration=self.asr_overlap_duration
        )
        return self._sentences(result)

    def _asr_sentences_from_array(self, audio_np, sample_rate):
        audio = audio_np.astype(np.float32)
        if sample_rate != self.asr_sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.asr_sample_rate)

        duration = len(audio) / self.asr_sample_rate
        if duration > self.long_audio_threshold_s:
            # Long recording: round-trip through a temp WAV so we can use parakeet's
            # tested long-form chunking (overlapping windows + stitched timestamps).
            import os
            import tempfile
            import soundfile as sf
            if self.debug:
                print(f"🎤 Long audio ({duration:.0f}s): chunked ASR via temp file")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            try:
                sf.write(tmp.name, audio, self.asr_sample_rate)
                result = self.asr_model.transcribe(
                    tmp.name,
                    chunk_duration=self.asr_chunk_duration,
                    overlap_duration=self.asr_overlap_duration,
                )
            finally:
                os.unlink(tmp.name)
            return self._sentences(result)

        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel
        mel = get_logmel(mx.array(audio), self.asr_model.preprocessor_config)
        result = self.asr_model.generate(mel)[0]
        return self._sentences(result)

    @staticmethod
    def _sentences(result):
        """Flatten an AlignedResult into [{text, start, end}, ...] of clean sentences."""
        sentences = []
        for s in result.sentences:
            text = s.text.strip()
            if text:
                sentences.append({"text": text, "start": float(s.start), "end": float(s.end)})
        return sentences

    # --- Diarization -----------------------------------------------------

    def _diarize_timeline(self, audio_source):
        """Return a sorted speaker timeline [{start, end, speaker}], or [] if no diarizer."""
        if self.diar_pipeline is None:
            return []
        try:
            output = self.diar_pipeline(audio_source)
        except Exception as e:
            # Never let a diarization failure throw away the transcript -- fall back
            # to ASR-only (single speaker) instead of losing everything.
            print(f"⚠️  Diarization failed at runtime ({e}); using single speaker.", file=sys.stderr)
            return []
        annotation = getattr(output, "speaker_diarization", output)
        timeline = [
            {"start": float(seg.start), "end": float(seg.end), "speaker": label}
            for seg, _, label in annotation.itertracks(yield_label=True)
        ]
        timeline.sort(key=lambda s: s["start"])
        if self.debug:
            speakers = sorted({s["speaker"] for s in timeline})
            print(f"🎯 {len(timeline)} segments across {len(speakers)} speakers: {speakers}")
        return timeline

    @staticmethod
    def _speaker_for_time(t, timeline):
        for seg in timeline:
            if seg["start"] <= t <= seg["end"]:
                return seg["speaker"]
        if not timeline:
            return None
        # Word sits in a diarization gap: attribute to the nearest segment.
        nearest = min(timeline, key=lambda seg: min(abs(t - seg["start"]), abs(t - seg["end"])))
        return nearest["speaker"]

    # --- Alignment -------------------------------------------------------

    def _assign_and_group(self, sentences, timeline):
        """Assign each sentence to a speaker by its midpoint, merging consecutive
        same-speaker sentences into turns. Returns segments shaped like the legacy
        pipeline: [{speaker, text, start_time, end_time}, ...]."""
        segments = []
        current = None
        for sent in sentences:
            midpoint = (sent["start"] + sent["end"]) / 2
            speaker = self._speaker_for_time(midpoint, timeline)
            speaker = speaker if speaker else "Speaker 1"

            if current and current["speaker"] == speaker:
                current["text"] += " " + sent["text"]
                current["end_time"] = sent["end"]
            else:
                if current:
                    segments.append(current)
                current = {
                    "speaker": speaker,
                    "text": sent["text"],
                    "start_time": sent["start"],
                    "end_time": sent["end"],
                }
        if current:
            segments.append(current)
        return segments

    # --- Public API ------------------------------------------------------

    def transcribe_array(self, audio_np, sample_rate):
        """In-memory path for live capture: float32 mono samples -> segments."""
        import torch
        sentences = self._asr_sentences_from_array(audio_np, sample_rate)
        if not sentences:
            return []
        waveform = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
        timeline = self._diarize_timeline({"waveform": waveform, "sample_rate": sample_rate})
        return self._assign_and_group(sentences, timeline)

    def transcribe_file(self, path):
        """File path -> segments."""
        import torchaudio
        sentences = self._asr_sentences_from_file(path)
        if not sentences:
            return []
        waveform, sample_rate = torchaudio.load(path)
        timeline = self._diarize_timeline({"waveform": waveform, "sample_rate": sample_rate})
        return self._assign_and_group(sentences, timeline)


def main():
    parser = argparse.ArgumentParser(
        description="Modern ASR + diarization engine (Parakeet-MLX + pyannote community-1)"
    )
    parser.add_argument("audio", help="Path to an audio file (wav/mp3/m4a/...)")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--diar-model", default=DEFAULT_DIAR_MODEL)
    parser.add_argument("--debug", action="store_true", help="Verbose progress output")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not found in environment / .env", file=sys.stderr)
        sys.exit(1)

    device = pick_device()
    if args.debug:
        print(f"🖥️  Device: {device}")

    engine = ModernEngine(args.asr_model, args.diar_model, hf_token, device, debug=args.debug)
    segments = engine.transcribe_file(args.audio)

    if not segments:
        print("No speech transcribed.", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 25 + " TRANSCRIPT " + "=" * 25)
    for seg in segments:
        print(f"[{seg['start_time']:0>7.2f}s] 👤 {seg['speaker']}: {seg['text']}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
