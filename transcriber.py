import os

import torch
from faster_whisper import WhisperModel

DEFAULT_PROMPT = (
    "This is a Hinglish medical conversation involving Hindi and English. "
    "Use words like 'achha', 'theek hai', 'doctor', 'patient', 'dawayi', "
    "'treatment', and 'checkup' when appropriate."
)

# Prevent the OpenMP runtime error on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class MedTranscriber:
    def __init__(
        self,
        model_size="large-v3",
        device=None,
        language="hi",
        beam_size=5,
        vad_filter=True,
        initial_prompt=DEFAULT_PROMPT,
    ):
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if resolved_device == "cuda" and not torch.cuda.is_available():
            print("--- CUDA requested but not available; falling back to CPU ---")
            resolved_device = "cpu"

        self.device = resolved_device
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.initial_prompt = initial_prompt
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(
            f"--- Loading Whisper {model_size} on "
            f"{self.device.upper()} ({self.compute_type}) ---"
        )

        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=max(1, os.cpu_count() or 4),
        )

    def transcribe(self, audio_path):
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=self.beam_size,
            language=self.language,
            initial_prompt=self.initial_prompt,
            vad_filter=self.vad_filter,
        )

        transcript_data = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            transcript_data.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": text,
                }
            )
            print(f"[{segment.start:.2f}s] {text}")

        return transcript_data, info.language
