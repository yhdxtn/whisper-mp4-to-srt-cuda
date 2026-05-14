import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import whisper
from whisper.utils import get_writer


VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".m4v"
}


def print_line(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def prepare_ffmpeg() -> Tuple[str, str]:
    """
    Return ffmpeg command/path and make sure Whisper can also find ffmpeg.
    Priority:
    1. System ffmpeg from PATH
    2. imageio-ffmpeg bundled ffmpeg
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        print("FFmpeg found:", system_ffmpeg)
        return "ffmpeg", system_ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError:
        print("ERROR: ffmpeg was not found, and imageio-ffmpeg is not installed.")
        print("Run this command first:")
        print("  python -m pip install imageio-ffmpeg")
        sys.exit(1)

    bundled = imageio_ffmpeg.get_ffmpeg_exe()
    if not bundled or not Path(bundled).exists():
        print("ERROR: imageio-ffmpeg is installed, but bundled ffmpeg was not found.")
        sys.exit(1)

    # Whisper internally calls command name "ffmpeg".
    # So we copy bundled ffmpeg to local .ffmpeg_bin/ffmpeg.exe and prepend it to PATH.
    bin_dir = Path(".ffmpeg_bin")
    bin_dir.mkdir(exist_ok=True)

    local_ffmpeg = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")

    try:
        if not local_ffmpeg.exists() or local_ffmpeg.stat().st_size != Path(bundled).stat().st_size:
            shutil.copy2(bundled, local_ffmpeg)
    except Exception:
        # If copy fails, still use the absolute bundled path for extraction,
        # but Whisper may still require PATH ffmpeg.
        pass

    if local_ffmpeg.exists():
        os.environ["PATH"] = str(bin_dir.resolve()) + os.pathsep + os.environ.get("PATH", "")
        print("Using bundled FFmpeg:", local_ffmpeg.resolve())
        return str(local_ffmpeg.resolve()), str(local_ffmpeg.resolve())

    print("Using bundled FFmpeg:", bundled)
    return bundled, bundled


def check_cuda(device):
    if device == "cuda":
        if torch.cuda.is_available():
            print("CUDA is available:", torch.cuda.get_device_name(0))
            return "cuda"
        print("WARNING: CUDA is not available. Falling back to CPU.")
        return "cpu"
    return "cpu"


def collect_videos(input_path):
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("Unsupported video file: {}".format(input_path))
        return [input_path]

    if input_path.is_dir():
        videos = [
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]
        return sorted(videos)

    raise FileNotFoundError("Input path does not exist: {}".format(input_path))


def extract_audio(ffmpeg_cmd, video_path, audio_path, audio_format):
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_format == "wav":
        cmd = [
            ffmpeg_cmd,
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            str(audio_path),
        ]
    else:
        cmd = [
            ffmpeg_cmd,
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "128k",
            str(audio_path),
        ]

    print("Extracting audio:", video_path.name)
    subprocess.run(cmd, check=True)


def transcribe_to_srt(
    model,
    audio_path,
    srt_dir,
    language,
    task,
    use_fp16,
):
    srt_dir.mkdir(parents=True, exist_ok=True)

    print("Transcribing:", audio_path.name)

    transcribe_kwargs = {
        "task": task,
        "fp16": use_fp16,
        "verbose": False,
    }

    if language and language.lower() not in {"auto", "none"}:
        transcribe_kwargs["language"] = language

    result = model.transcribe(str(audio_path), **transcribe_kwargs)

    writer = get_writer("srt", str(srt_dir))
    writer(result, str(audio_path))

    srt_path = srt_dir / "{}.srt".format(audio_path.stem)
    return srt_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert videos to SRT subtitles using Whisper with CUDA support."
    )

    parser.add_argument("--input", default="videos", help="Input video file or folder. Default: videos")
    parser.add_argument("--audio-dir", default="audio", help="Output folder for extracted audio. Default: audio")
    parser.add_argument("--srt-dir", default="subtitles", help="Output folder for SRT subtitles. Default: subtitles")
    parser.add_argument("--model", default="medium", help="Whisper model: tiny, base, small, medium, large, turbo. Default: medium")
    parser.add_argument("--language", default="Chinese", help="Language, for example Chinese, English. Use auto for auto-detect. Default: Chinese")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device: cuda or cpu. Default: cuda")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="transcribe keeps original language; translate outputs English. Default: transcribe")
    parser.add_argument("--audio-format", default="wav", choices=["wav", "mp3"], help="Intermediate audio format. Default: wav")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos when matching SRT already exists.")

    args = parser.parse_args()

    ffmpeg_cmd, ffmpeg_path = prepare_ffmpeg()

    input_path = Path(args.input)
    audio_dir = Path(args.audio_dir)
    srt_dir = Path(args.srt_dir)

    Path("videos").mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    srt_dir.mkdir(exist_ok=True)

    device = check_cuda(args.device)
    use_fp16 = device == "cuda"

    videos = collect_videos(input_path)

    if not videos:
        print("No video files found.")
        print("You can run with a specific video path:")
        print('  python transcribe.py --input "D:\\your_video.mp4"')
        return

    print_line("Loading Whisper model")
    print("Model:", args.model)
    print("Device:", device)
    print("Language:", args.language)
    model = whisper.load_model(args.model, device=device)

    print_line("Starting")
    print("Found {} video file(s).".format(len(videos)))

    success_count = 0
    failed_count = 0

    for index, video_path in enumerate(videos, start=1):
        print_line("[{}/{}] {}".format(index, len(videos), video_path.name))

        audio_path = audio_dir / "{}.{}".format(video_path.stem, args.audio_format)
        srt_path = srt_dir / "{}.srt".format(video_path.stem)

        if args.skip_existing and srt_path.exists():
            print("Skipped because SRT already exists:", srt_path)
            continue

        try:
            extract_audio(ffmpeg_cmd, video_path, audio_path, args.audio_format)
            generated_srt = transcribe_to_srt(
                model=model,
                audio_path=audio_path,
                srt_dir=srt_dir,
                language=args.language,
                task=args.task,
                use_fp16=use_fp16,
            )
            print("Done:", generated_srt)
            success_count += 1
        except subprocess.CalledProcessError as e:
            print("FFmpeg failed for:", video_path)
            print(e)
            failed_count += 1
        except Exception as e:
            print("Failed for:", video_path)
            print(e)
            failed_count += 1

    print_line("Finished")
    print("Success:", success_count)
    print("Failed:", failed_count)
    print("SRT folder:", srt_dir.resolve())


if __name__ == "__main__":
    main()
