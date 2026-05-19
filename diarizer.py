import os
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()


class Diarizer:
    def __init__(self, device=None, token=None):
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "cuda" and not torch.cuda.is_available():
            print("--- CUDA requested for diarization but not available; using CPU ---")
            requested_device = "cpu"

        self.device = torch.device(requested_device)
        token = token or os.getenv("HF_TOKEN")

        if not token:
            raise ValueError(
                "HF_TOKEN not found in .env file. Run without diarization or add a token."
            )

        print(f"--- Loading Pyannote Diarization on {self.device} ---")

        # Load the gated model from HuggingFace
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )

        if self.pipeline:
            self.pipeline.to(self.device)
        else:
            raise Exception("Failed to initialize Pyannote pipeline.")

    def apply(self, audio_path):
        print("--- Analyzing Speaker Timestamps... ---")
        diarization = self.pipeline(audio_path)
        
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        speaker_segments.sort(key=lambda item: (item["start"], item["end"]))
        return speaker_segments
