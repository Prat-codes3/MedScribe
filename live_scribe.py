import argparse
import os
import queue
import sys
import time
from dataclasses import dataclass
from datetime import datetime

DEFAULT_PROMPT = (
    "Hinglish medical conversation: Hindi and English mixed. "
    "Examples include 'theek hai', 'doctor', 'patient', 'checkup', "
    "'treatment', 'dawayi', and 'bukhaar'."
)
INPUT_BLOCK_SECONDS = 0.5
MAX_QUEUE_SIZE = 64
SAMPLE_RATE = 16000

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)


@dataclass
class LiveScribeConfig:
    model_size: str
    language: str
    device: str
    interval_seconds: float
    overlap_seconds: float
    beam_size: int
    use_vad: bool
    output_path: str | None = None


def resolve_device(device_choice):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'torch'. Install requirements before running live transcription."
        ) from exc

    if device_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_choice == "cuda" and not torch.cuda.is_available():
        print("--- CUDA requested but not available; falling back to CPU ---")
        return "cpu"
    return device_choice


def format_timestamp(seconds):
    minutes, secs = divmod(int(seconds), 60)
    return f"[{minutes:02d}:{secs:02d}]"


def build_output_path(output_path):
    if output_path:
        return os.path.abspath(output_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(f"live_session_{timestamp}.txt")


def audio_callback(indata, frames, time_info, status):
    """Capture audio chunks from the microphone without blocking the input stream."""
    if status:
        print(status, file=sys.stderr)

    chunk = indata.copy()

    try:
        audio_queue.put_nowait(chunk)
    except queue.Full:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
        audio_queue.put_nowait(chunk)


def load_model(config):
    try:
        import torch
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing live transcription dependencies. Run 'pip install -r requirements.txt'."
        ) from exc

    device = resolve_device(config.device)
    compute_type = "float16" if device == "cuda" else "int8"

    print(
        f"--- Loading Live Whisper {config.model_size} on "
        f"{device.upper()} ({compute_type}) ---"
    )

    model = WhisperModel(
        config.model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=max(1, os.cpu_count() or 4),
    )
    return model, device


def emit_segments(writer, session_start, window_base_seconds, overlap_seconds, segments, last_text):
    """Write only segments that belong to newly recorded audio."""
    for segment in segments:
        if segment.end <= overlap_seconds:
            continue

        text = segment.text.strip()
        if not text or text == last_text:
            continue

        absolute_start = window_base_seconds + max(segment.start, overlap_seconds)
        time_tag = format_timestamp(absolute_start)
        line = f"{time_tag} {text}"

        elapsed = time.time() - session_start
        sys.stdout.write(f"\r{format_timestamp(elapsed)} {text}   ")
        sys.stdout.flush()

        writer.write(f"{line}\n")
        writer.flush()
        last_text = text

    return last_text


def transcribe_window(
    model,
    config,
    writer,
    session_start,
    context_audio,
    new_audio,
    processed_seconds,
    last_text,
):
    import numpy as np

    audio_window = (
        new_audio
        if context_audio.size == 0
        else np.concatenate([context_audio, new_audio])
    ).astype(np.float32, copy=False)

    overlap_seconds = len(context_audio) / SAMPLE_RATE
    window_base_seconds = max(0.0, processed_seconds - overlap_seconds)

    segments, _ = model.transcribe(
        audio_window,
        beam_size=config.beam_size,
        language=config.language,
        initial_prompt=DEFAULT_PROMPT,
        vad_filter=config.use_vad,
        condition_on_previous_text=False,
    )
    return emit_segments(
        writer, session_start, window_base_seconds, overlap_seconds, segments, last_text
    )


def start_live_scribe(config):
    try:
        import numpy as np
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing live transcription dependencies. Run 'pip install -r requirements.txt'."
        ) from exc

    if config.overlap_seconds >= config.interval_seconds:
        raise ValueError("Overlap must be smaller than the transcription interval.")

    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    model, device = load_model(config)
    output_path = build_output_path(config.output_path)
    session_start = time.time()
    input_blocksize = int(SAMPLE_RATE * INPUT_BLOCK_SECONDS)
    interval_samples = int(SAMPLE_RATE * config.interval_seconds)
    overlap_samples = int(SAMPLE_RATE * config.overlap_seconds)

    print("\n--- LIVE HINGLISH SESSION STARTING ---")
    print(f"Device: {device.upper()}")
    print(f"Saving to: {output_path}")
    print("Speak now... (Press Ctrl+C to stop)")
    print("-" * 40)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=input_blocksize,
        dtype="float32",
    ):
        context_audio = np.empty(0, dtype=np.float32)
        pending_chunks = []
        pending_samples = 0
        processed_seconds = 0.0
        last_emitted_text = None

        with open(output_path, "w", encoding="utf-8") as writer:
            writer.write(
                "MEDSCRIBE LIVE SESSION\n"
                f"Started: {datetime.now()}\n"
                f"Model: {config.model_size}\n"
                f"Language: {config.language}\n"
                + "=" * 30
                + "\n\n"
            )
            writer.flush()

            try:
                while True:
                    chunk = audio_queue.get()
                    flattened_chunk = chunk.flatten().astype(np.float32, copy=False)
                    pending_chunks.append(flattened_chunk)
                    pending_samples += len(flattened_chunk)

                    if pending_samples < interval_samples:
                        continue

                    new_audio = np.concatenate(pending_chunks)
                    last_emitted_text = transcribe_window(
                        model,
                        config,
                        writer,
                        session_start,
                        context_audio,
                        new_audio,
                        processed_seconds,
                        last_emitted_text,
                    )

                    processed_seconds += len(new_audio) / SAMPLE_RATE
                    context_audio = (
                        new_audio[-overlap_samples:]
                        if overlap_samples
                        else np.empty(0, dtype=np.float32)
                    )
                    pending_chunks = []
                    pending_samples = 0

            except KeyboardInterrupt:
                if pending_chunks:
                    new_audio = np.concatenate(pending_chunks)
                    last_emitted_text = transcribe_window(
                        model,
                        config,
                        writer,
                        session_start,
                        context_audio,
                        new_audio,
                        processed_seconds,
                        last_emitted_text,
                    )

                writer.write("\n" + "=" * 30 + f"\nEnd Time: {datetime.now()}\n")
                writer.flush()
                print(f"\n\n--- SESSION ENDED & SAVED TO: {output_path} ---")


def parse_args():
    parser = argparse.ArgumentParser(description="Live MedScribe transcription")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MEDSCRIBE_LIVE_MODEL", "small"),
        help="Whisper model size for live transcription.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=os.getenv("MEDSCRIBE_LIVE_LANGUAGE", "hi"),
        help="Language hint to pass to Whisper.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=os.getenv("MEDSCRIBE_DEVICE", "auto"),
        help="Inference device to use.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds of new audio to collect before each transcription pass.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.75,
        help="Seconds of trailing context to keep between passes.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam size for faster-whisper.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD filtering.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for the saved live transcript.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_live_scribe(
        LiveScribeConfig(
            model_size=args.model,
            language=args.language,
            device=args.device,
            interval_seconds=args.interval,
            overlap_seconds=args.overlap,
            beam_size=args.beam_size,
            use_vad=not args.no_vad,
            output_path=args.output,
        )
    )
