import argparse
import subprocess
from pathlib import Path

from core import (
    prepare_ffmpeg,
    collect_videos,
    load_model,
    transcribe_video_to_srt,
    print_line,
)


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
    parser.add_argument("--cleanup-audio", action="store_true", help="Delete extracted audio after SRT is generated.")

    args = parser.parse_args()

    ffmpeg_cmd, ffmpeg_path = prepare_ffmpeg()

    input_path = Path(args.input)
    audio_dir = Path(args.audio_dir)
    srt_dir = Path(args.srt_dir)

    Path("videos").mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    srt_dir.mkdir(exist_ok=True)

    videos = collect_videos(input_path)

    if not videos:
        print("No video files found.")
        print("You can run with a specific video path:")
        print('  python transcribe.py --input "D:\\your_video.mp4"')
        return

    print_line("Loading Whisper model")
    print("Model:", args.model)
    print("Device:", args.device)
    print("Language:", args.language)

    model, device = load_model(args.model, args.device)

    print_line("Starting")
    print("Found {} video file(s).".format(len(videos)))

    success_count = 0
    failed_count = 0

    for index, video_path in enumerate(videos, start=1):
        print_line("[{}/{}] {}".format(index, len(videos), video_path.name))

        srt_path = srt_dir / "{}.srt".format(video_path.stem)

        if args.skip_existing and srt_path.exists():
            print("Skipped because SRT already exists:", srt_path)
            continue

        try:
            generated_srt = transcribe_video_to_srt(
                video_path=video_path,
                model=model,
                ffmpeg_cmd=ffmpeg_cmd,
                audio_dir=audio_dir,
                srt_dir=srt_dir,
                language=args.language,
                task=args.task,
                device=device,
                audio_format=args.audio_format,
                cleanup_audio=args.cleanup_audio,
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
