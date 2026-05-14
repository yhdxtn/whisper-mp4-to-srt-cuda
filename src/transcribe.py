\
import argparse
import shutil
import subprocess
from pathlib import Path

import torch
import whisper
from whisper.utils import get_writer


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".m4v"}


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "未检测到 ffmpeg。请先安装 FFmpeg，并确保 ffmpeg 可以在 CMD 中直接运行。"
        )


def check_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            print("警告：当前 PyTorch 没有检测到 CUDA，已自动切换到 CPU。")
            return "cpu"
        print(f"CUDA 可用：{torch.cuda.get_device_name(0)}")
    return device


def collect_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"输入文件不是支持的视频格式：{input_path}")
        return [input_path]

    if input_path.is_dir():
        videos = [
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]
        return sorted(videos)

    raise FileNotFoundError(f"输入路径不存在：{input_path}")


def extract_audio(video_path: Path, audio_path: Path, audio_format: str) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_format == "wav":
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            str(audio_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "128k",
            str(audio_path),
        ]

    print(f"\n正在提取音频：{video_path.name}")
    subprocess.run(cmd, check=True)


def transcribe_audio(
    model,
    audio_path: Path,
    srt_dir: Path,
    language: str,
    task: str,
) -> None:
    srt_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在识别字幕：{audio_path.name}")

    result = model.transcribe(
        str(audio_path),
        language=language,
        task=task,
        verbose=False,
        fp16=torch.cuda.is_available(),
    )

    writer = get_writer("srt", str(srt_dir))
    writer(result, str(audio_path))

    generated_srt = srt_dir / f"{audio_path.stem}.srt"
    print(f"字幕已生成：{generated_srt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 Whisper + CUDA 将 MP4 视频批量转换为 SRT 字幕"
    )
    parser.add_argument("--input", default="videos", help="输入视频文件或视频文件夹")
    parser.add_argument("--audio-dir", default="audio", help="音频输出目录")
    parser.add_argument("--srt-dir", default="subtitles", help="字幕输出目录")
    parser.add_argument("--model", default="medium", help="Whisper 模型：tiny/base/small/medium/large/turbo")
    parser.add_argument("--language", default="Chinese", help="语言，例如 Chinese、English、Japanese")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="运行设备")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="转录或翻译")
    parser.add_argument("--audio-format", default="wav", choices=["wav", "mp3"], help="中间音频格式")
    parser.add_argument("--skip-existing", action="store_true", help="如果 srt 已存在，则跳过")
    args = parser.parse_args()

    check_ffmpeg()

    input_path = Path(args.input)
    audio_dir = Path(args.audio_dir)
    srt_dir = Path(args.srt_dir)

    videos = collect_videos(input_path)
    if not videos:
        print(f"没有找到视频文件，请把 mp4 放入：{input_path.resolve()}")
        return

    device = check_device(args.device)

    print(f"\n正在加载 Whisper 模型：{args.model}")
    model = whisper.load_model(args.model, device=device)

    print(f"共找到 {len(videos)} 个视频。")

    for index, video_path in enumerate(videos, start=1):
        print(f"\n========== [{index}/{len(videos)}] {video_path.name} ==========")

        audio_path = audio_dir / f"{video_path.stem}.{args.audio_format}"
        srt_path = srt_dir / f"{video_path.stem}.srt"

        if args.skip_existing and srt_path.exists():
            print(f"字幕已存在，跳过：{srt_path}")
            continue

        extract_audio(video_path, audio_path, args.audio_format)
        transcribe_audio(
            model=model,
            audio_path=audio_path,
            srt_dir=srt_dir,
            language=args.language,
            task=args.task,
        )

    print("\n全部处理完成。")


if __name__ == "__main__":
    main()
