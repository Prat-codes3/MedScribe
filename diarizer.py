from dotenv import load_dotenv
load_dotenv()
from pyannote.audio import Pipeline
import torch
import os

class Diarizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token = os.getenv("HF_TOKEN")
        
        print(f"--- Loading Pyannote on {self.device} ---")
        
        # This will now use your token to download the model
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        
        if self.pipeline is None:
            raise ValueError("Pipeline failed to load. Check your HF_TOKEN and permissions.")
            
        self.pipeline.to(self.device)

    def diarize(self, audio_path):
        """Returns a list of speaker segments."""
        print("Running diarization... This may take a moment.")
        diarization = self.pipeline(audio_path)
        
        speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return speakers