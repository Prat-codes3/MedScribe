import argparse
import os
from dotenv import load_dotenv
from transcriber import Transcriber
from diarizer import Diarizer
from utils import record_audio_chunk, save_temp_audio

load_dotenv()

def align_transcripts_with_speakers(transcripts, diarization):
    """Maps text segments to speakers based on overlapping timestamps."""
    final_output = []
    
    for t_seg in transcripts:
        t_start, t_end = t_seg["start"], t_seg["end"]
        best_speaker = "SPEAKER_UNKNOWN"
        max_overlap = 0
        
        for d_seg in diarization:
            # Calculate overlap duration
            overlap_start = max(t_start, d_seg["start"])
            overlap_end = min(t_end, d_seg["end"])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = d_seg["speaker"]
                
        final_output.append(f"[{t_start:.2f} - {t_end:.2f}] {best_speaker}: {t_seg['text']}")
        
    return "\n".join(final_output)

def run_file_mode(file_path, mode, model_size):
    transcriber = Transcriber(model_size=model_size)
    transcripts, lang = transcriber.transcribe(file_path)
    
    print(f"Detected Language: {lang}")
    
    if mode == "fast":
        output = "\n".join([f"[{s['start']:.2f} - {s['end']:.2f}]: {s['text']}" for s in transcripts])
    else:
        diarizer = Diarizer()
        diarization = diarizer.diarize(file_path)
        output = align_transcripts_with_speakers(transcripts, diarization)
        
    print("\n--- FINAL TRANSCRIPT ---\n")
    print(output)
    
    with open("transcript_output.txt", "w", encoding="utf-8") as f:
        f.write(output)
    print("\nSaved to transcript_output.txt")

def run_live_mode(mode, model_size, chunk_duration=5):
    """
    Real-time transcription.
    Note: Real-time diarization is mathematically unstable on short chunks.
    This runs live transcription, and optionally diarizes the whole session upon stopping.
    """
    transcriber = Transcriber(model_size=model_size)
    print("\nStarting LIVE mode. Press Ctrl+C to stop.\n")
    
    session_audio = []
    fs = 16000
    
    try:
        while True:
            chunk, _ = record_audio_chunk(duration=chunk_duration, fs=fs)
            session_audio.append(chunk)
            
            # Process chunk immediately for live feedback
            temp_path = save_temp_audio(chunk, fs)
            transcripts, _ = transcriber.transcribe(temp_path)
            for t in transcripts:
                print(f"LIVE: {t['text']}")
            os.remove(temp_path)
            
    except KeyboardInterrupt:
        print("\nLive recording stopped.")
        
        # Combine all chunks for final processing
        if session_audio:
            full_audio = np.concatenate(session_audio)
            final_path = save_temp_audio(full_audio, fs)
            
            print("Processing final complete transcript...")
            run_file_mode(final_path, mode, model_size)
            os.remove(final_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Transcription & Diarization")
    parser.add_argument("--source", type=str, choices=["live", "file"], required=True, help="Input source")
    parser.add_argument("--file", type=str, help="Path to audio/video file (if source=file)")
    parser.add_argument("--mode", type=str, choices=["fast", "full"], default="fast", help="'fast' (no diarization) or 'full' (with diarization)")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model size (tiny, base, small, medium, large-v3)")
    
    args = parser.parse_args()
    
    if args.source == "file":
        if not args.file:
            print("Error: --file argument is required when source is 'file'")
        else:
            run_file_mode(args.file, args.mode, args.model)
    elif args.source == "live":
        run_live_mode(args.mode, args.model)