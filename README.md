# MedScribe

MedScribe is a small Python CLI project for transcribing medical or Hinglish audio.
It supports two main workflows:

- Batch transcription from an audio file with optional speaker diarization.
- Live microphone transcription tuned for low-latency rolling updates.

## Project Layout

- `main.py` runs file-based transcription and saves a transcript next to the source audio.
- `transcriber.py` wraps `faster-whisper` and applies Hinglish-friendly prompting.
- `diarizer.py` wraps `pyannote.audio` for optional speaker labeling.
- `live_scribe.py` listens to the microphone and transcribes in rolling windows.
- `proto.py` and `utils.py` are older prototypes kept for reference.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env`.
4. Add your Hugging Face token to `.env` if you want diarization.

## Usage(to run)

Batch transcription:

```bash

python main.py --file path/to/audio.wav
python main.py --file path/to/audio.wav --diarize
python main.py --file path/to/audio.wav --model medium --device cpu
```

Batch transcription defaults to Hindi. Override it when needed:

```bash
python main.py --file path/to/audio.wav --language en
```

Live transcription:

```bash
python live_scribe.py
python live_scribe.py --model small --interval 2.0 --overlap 0.75
python live_scribe.py --device cpu --output session.txt
```

## Notes

- Diarization is slower and requires `HF_TOKEN`.
- The live pipeline now transcribes only new audio plus a short overlap window, which is much faster than reprocessing the entire session buffer each time.
- `.gitignore` excludes generated media, transcript outputs, and environment files.



