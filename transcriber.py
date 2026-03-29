import os
# Fix for the OpenMP error we just saw
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel
import torch

class Transcriber:
    def __init__(self, model_size="base"):
        # Detect if CUDA (GPU) is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"--- Loading Whisper model '{model_size}' on {device.upper()} ---")
        
        # float16 is best for your RTX 4050 (fast + low VRAM usage)
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type="float16" if device == "cuda" else "int8"
        )
    def transcribe(self, audio_path):
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        
        print(f"--- Detected language: {info.language} ---")

        full_text = ""
        for segment in segments:
            # This is what you see printing in your terminal
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            full_text += segment.text + " "
            
        # FIX: Return TWO values (text and language) to satisfy main.py
        return full_text.strip(), info.language

if __name__ == "__main__":
    # Quick test if you have a wav file handy
    # processor = MedTranscriber()
    # processor.transcribe("test_audio.wav")
    pass