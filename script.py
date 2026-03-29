import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import argparse
import os
import queue
import threading

# Load model
model = whisper.load_model("base")  # or "small"

samplerate = 16000
q = queue.Queue()

# 🎤 Callback for continuous audio
def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

# 🎤 Continuous Live Transcription
def live_transcription():
    print("🎤 Continuous Listening... (Press Ctrl+C to stop)\n")

    with sd.InputStream(samplerate=samplerate,
                        channels=1,
                        dtype='float32',
                        callback=audio_callback):

        audio_buffer = []

        while True:
            try:
                data = q.get()
                audio_buffer.append(data)

                # Process every ~3 seconds
                if len(audio_buffer) >= int(samplerate / 1024 * 3):
                    audio_np = np.concatenate(audio_buffer, axis=0)

                    filename = "temp_live.wav"
                    wav.write(filename, samplerate, audio_np)

                    result = model.transcribe(
                        filename,
                        language="hi"   # 🔥 Hinglish trick
                    )

                    print("📝", result["text"])

                    os.remove(filename)
                    audio_buffer = []

            except KeyboardInterrupt:
                print("\n🛑 Stopped listening.")
                break


# 📁 File Transcription
def transcribe_file(filepath):
    if not os.path.exists(filepath):
        print("❌ File not found!")
        return

    print(f"🧠 Transcribing file: {filepath}")

    result = model.transcribe(
        filepath,
        language="hi"   # Hinglish support
    )

    print("📝 Result:", result["text"])


# 🎯 Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True,
                        help="live or file")

    parser.add_argument("--file", type=str,
                        help="audio file path")

    args = parser.parse_args()

    if args.mode == "live":
        live_transcription()

    elif args.mode == "file":
        if args.file is None:
            print("❌ Provide file path using --file")
        else:
            transcribe_file(args.file)

    else:
        print("❌ Invalid mode! Use 'live' or 'file'")