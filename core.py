import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import whisper
from whisper.utils import get_writer


VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".m4v", ".webm"
}


def print_line(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def prepare_ffmpeg() -> Tuple[str, str]:
    """
    Return ffmpeg command/path and make sure Whisper and yt-dlp can find ffmpeg.

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

    bin_dir = Path(".ffmpeg_bin")
    bin_dir.mkdir(exist_ok=True)

    local_ffmpeg = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")

    try:
        if not local_ffmpeg.exists() or local_ffmpeg.stat().st_size != Path(bundled).stat().st_size:
            shutil.copy2(bundled, local_ffmpeg)
    except Exception:
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
    input_path = Path(input_path)

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
    video_path = Path(video_path)
    audio_path = Path(audio_path)
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


def load_model(model_name, device):
    device = check_cuda(device)
    print("Loading Whisper model:", model_name)
    model = whisper.load_model(model_name, device=device)
    return model, device


def transcribe_audio_to_srt(
    model,
    audio_path,
    srt_dir,
    language,
    task,
    use_fp16,
):
    audio_path = Path(audio_path)
    srt_dir = Path(srt_dir)
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

    return srt_dir / "{}.srt".format(audio_path.stem)


def transcribe_video_to_srt(
    video_path,
    model,
    ffmpeg_cmd,
    audio_dir,
    srt_dir,
    language="Chinese",
    task="transcribe",
    device="cuda",
    audio_format="wav",
    cleanup_audio=False,
    output_stem=None,
):
    video_path = Path(video_path)
    audio_dir = Path(audio_dir)
    srt_dir = Path(srt_dir)

    stem = output_stem or video_path.stem
    audio_path = audio_dir / "{}.{}".format(stem, audio_format)

    extract_audio(ffmpeg_cmd, video_path, audio_path, audio_format)

    srt_path = transcribe_audio_to_srt(
        model=model,
        audio_path=audio_path,
        srt_dir=srt_dir,
        language=language,
        task=task,
        use_fp16=(device == "cuda"),
    )

    if cleanup_audio:
        try:
            audio_path.unlink(missing_ok=True)
        except TypeError:
            if audio_path.exists():
                audio_path.unlink()
        except Exception:
            pass

    return srt_path


def download_bilibili_video(url, download_dir, ffmpeg_path=None):
    """
    Download one Bilibili video with yt-dlp and return local video path.

    Some Bilibili videos may require login cookies or may fail when the site changes.
    For public videos, yt-dlp usually works without cookies.
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise RuntimeError("yt-dlp is not installed. Run: python -m pip install yt-dlp")

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_location = None
    if ffmpeg_path and ffmpeg_path != "ffmpeg":
        p = Path(ffmpeg_path)
        ffmpeg_location = str(p.parent if p.is_file() else p)

    outtmpl = str(download_dir / "%(title).80s-%(id)s.%(ext)s")

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bv*+ba/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "restrictfilenames": True,
    }

    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location

    before = set(download_dir.glob("*"))

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    after = set(download_dir.glob("*"))
    new_files = [p for p in (after - before) if p.is_file()]

    candidates = [
        p for p in download_dir.glob("*")
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not p.name.endswith(".part")
    ]

    if new_files:
        fresh_candidates = [
            p for p in new_files
            if p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.endswith(".part")
        ]
        if fresh_candidates:
            candidates = fresh_candidates

    if not candidates:
        raise RuntimeError("Bilibili video download finished, but no video file was found.")

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def safe_delete_path(path):
    path = Path(path)
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(str(path), ignore_errors=True)
    except Exception:
        pass


def make_job_id():
    return time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
