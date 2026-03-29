import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tempfile
import os

def record_audio_chunk(duration=5, fs=16000):
    """Records a chunk of audio from the microphone."""
    print(f"🎤 Recording {duration}s chunk...")
    # Record in Mono (channels=1)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    return recording, fs

def save_temp_audio(recording, fs):
    """Saves numpy array to a temporary wav file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, fs, recording)
    return temp_file.name