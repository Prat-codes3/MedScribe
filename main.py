import argparse
import os
from datetime import datetime


def format_timestamp(seconds):
    minutes, secs = divmod(int(seconds), 60)
    return f"[{minutes:02d}:{secs:02d}]"


def find_speaker(segment, speaker_data, search_start=0):
    """Match a transcript segment to the diarization segment with the largest overlap."""
    segment_start = segment["start"]
    segment_end = segment["end"]
    midpoint = (segment_start + segment_end) / 2

    while search_start < len(speaker_data) and speaker_data[search_start]["end"] < segment_start:
        search_start += 1

    best_speaker = "SPEAKER_UNKNOWN"
    best_overlap = 0.0
    cursor = search_start

    while cursor < len(speaker_data) and speaker_data[cursor]["start"] <= segment_end:
        diarized_segment = speaker_data[cursor]
        overlap = min(segment_end, diarized_segment["end"]) - max(
            segment_start, diarized_segment["start"]
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = diarized_segment["speaker"]
        elif (
            best_overlap == 0
            and diarized_segment["start"] <= midpoint <= diarized_segment["end"]
        ):
            best_speaker = diarized_segment["speaker"]

        cursor += 1

    return best_speaker, search_start


def build_transcript(file_path, language, text_segments, speaker_data):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    diarization_status = "enabled" if speaker_data else "disabled"

    final_lines = [
        "MEDSCRIBE AI TRANSCRIPT\n",
        f"Source File: {file_path}\n",
        f"Processed on: {timestamp_str}\n",
        f"Language: {language}\n",
        f"Diarization: {diarization_status}\n",
        "-" * 30 + "\n",
    ]

    current_speaker = None
    speaker_index = 0

    for segment in text_segments:
        speaker = None
        if speaker_data:
            speaker, speaker_index = find_speaker(segment, speaker_data, speaker_index)

        time_tag = format_timestamp(segment["start"])
        text = segment["text"]

        if speaker_data:
            if speaker != current_speaker:
                line = f"\n{time_tag} {speaker}: {text}\n"
                current_speaker = speaker
            else:
                line = f"{text}\n"
        else:
            line = f"{time_tag} {text}\n"

        print(line.rstrip())
        final_lines.append(line)

    return final_lines


def process_audio(file_path, model_name, diarize=False, device=None, language="hi"):
    from diarizer import Diarizer
    from transcriber import MedTranscriber

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    absolute_input_path = os.path.abspath(file_path)

    print("\n--- Phase 1: Transcription ---")
    transcriber = MedTranscriber(model_size=model_name, device=device, language=language)
    text_segments, detected_language = transcriber.transcribe(absolute_input_path)

    speaker_data = []
    if diarize:
        print("\n--- Phase 2: Speaker ID ---")
        try:
            diarizer = Diarizer(device=device)
            speaker_data = diarizer.apply(absolute_input_path)
        except Exception as exc:
            print(f"Warning: speaker diarization unavailable ({exc}). Continuing without speakers.")

    print("\n--- FINAL SCRIBE REPORT ---")
    final_lines = build_transcript(
        absolute_input_path, detected_language, text_segments, speaker_data
    )

    base_name = os.path.splitext(os.path.basename(absolute_input_path))[0]
    output_filename = os.path.join(
        os.path.dirname(absolute_input_path), f"{base_name}_transcript.txt"
    )

    with open(output_filename, "w", encoding="utf-8") as f:
        f.writelines(final_lines)

    print(f"\n\n--- SUCCESS ---")
    print(f"Full transcript saved automatically to: {os.path.abspath(output_filename)}")


def resolve_device(device_choice):
    if device_choice == "auto":
        return None
    return device_choice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedScribe AI")
    parser.add_argument("--file", type=str, required=True, help="Path to audio file")
    parser.add_argument("--model", type=str, default="large-v3", help="Whisper model size")
    parser.add_argument(
        "--language",
        type=str,
        default="hi",
        help="Language hint for Whisper. Defaults to Hindi.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device to use.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable slower speaker diarization. Requires HF_TOKEN.",
    )

    args = parser.parse_args()
    process_audio(
        args.file,
        args.model,
        diarize=args.diarize,
        device=resolve_device(args.device),
        language=args.language,
    )
