from pathlib import Path

from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

from core import (
    VIDEO_EXTENSIONS,
    prepare_ffmpeg,
    load_model,
    transcribe_video_to_srt,
    download_bilibili_video,
    safe_delete_path,
    make_job_id,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DOWNLOAD_DIR = BASE_DIR / "downloads"
AUDIO_DIR = BASE_DIR / "audio"
SRT_DIR = BASE_DIR / "subtitles"

for folder in [UPLOAD_DIR, DOWNLOAD_DIR, AUDIO_DIR, SRT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4096 * 1024 * 1024

MODEL_CACHE = {}


def is_local_request():
    remote = request.remote_addr or ""
    return remote in {"127.0.0.1", "::1", "localhost"}


def get_cached_model(model_name, device):
    key = "{}:{}".format(model_name, device)
    if key not in MODEL_CACHE:
        model, real_device = load_model(model_name, device)
        MODEL_CACHE[key] = (model, real_device)
    return MODEL_CACHE[key]


def allowed_video(filename):
    return Path(filename).suffix.lower() in VIDEO_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        is_local=is_local_request(),
        default_model="medium",
        default_language="Chinese",
    )


@app.route("/process", methods=["POST"])
def process():
    source_type = request.form.get("source_type", "upload")
    model_name = request.form.get("model", "medium").strip() or "medium"
    language = request.form.get("language", "Chinese").strip() or "Chinese"
    device = request.form.get("device", "cuda").strip() or "cuda"
    audio_format = request.form.get("audio_format", "wav").strip() or "wav"

    job_id = make_job_id()
    job_upload_dir = UPLOAD_DIR / job_id
    job_download_dir = DOWNLOAD_DIR / job_id
    job_audio_dir = AUDIO_DIR / job_id

    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_download_dir.mkdir(parents=True, exist_ok=True)
    job_audio_dir.mkdir(parents=True, exist_ok=True)

    downloaded_video = None
    uploaded_video = None

    try:
        ffmpeg_cmd, ffmpeg_path = prepare_ffmpeg()
        model, real_device = get_cached_model(model_name, device)

        if source_type == "local":
            if not is_local_request():
                raise RuntimeError("当前不是本机访问，不能使用服务器本地路径。请改为上传视频或输入哔哩哔哩链接。")

            local_path = request.form.get("local_path", "").strip().strip('"')
            if not local_path:
                raise RuntimeError("请输入本地视频路径。")

            video_path = Path(local_path)
            if not video_path.exists() or not video_path.is_file():
                raise RuntimeError("本地视频不存在：{}".format(video_path))

            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise RuntimeError("不支持的视频格式：{}".format(video_path.suffix))

            output_stem = video_path.stem

            srt_path = transcribe_video_to_srt(
                video_path=video_path,
                model=model,
                ffmpeg_cmd=ffmpeg_cmd,
                audio_dir=job_audio_dir,
                srt_dir=SRT_DIR,
                language=language,
                task="transcribe",
                device=real_device,
                audio_format=audio_format,
                cleanup_audio=True,
                output_stem=output_stem,
            )

            message = "本地视频处理完成，原视频不会删除。"

        elif source_type == "upload":
            file = request.files.get("video_file")
            if not file or not file.filename:
                raise RuntimeError("请选择要上传的视频文件。")

            if not allowed_video(file.filename):
                raise RuntimeError("不支持的视频格式。")

            filename = secure_filename(file.filename)
            uploaded_video = job_upload_dir / filename
            file.save(str(uploaded_video))

            output_stem = uploaded_video.stem

            srt_path = transcribe_video_to_srt(
                video_path=uploaded_video,
                model=model,
                ffmpeg_cmd=ffmpeg_cmd,
                audio_dir=job_audio_dir,
                srt_dir=SRT_DIR,
                language=language,
                task="transcribe",
                device=real_device,
                audio_format=audio_format,
                cleanup_audio=True,
                output_stem=output_stem,
            )

            message = "上传视频处理完成。"

        elif source_type == "bilibili":
            bilibili_url = request.form.get("bilibili_url", "").strip()
            if not bilibili_url:
                raise RuntimeError("请输入哔哩哔哩视频地址。")

            downloaded_video = download_bilibili_video(
                bilibili_url,
                job_download_dir,
                ffmpeg_path=ffmpeg_path,
            )

            output_stem = downloaded_video.stem

            srt_path = transcribe_video_to_srt(
                video_path=downloaded_video,
                model=model,
                ffmpeg_cmd=ffmpeg_cmd,
                audio_dir=job_audio_dir,
                srt_dir=SRT_DIR,
                language=language,
                task="transcribe",
                device=real_device,
                audio_format=audio_format,
                cleanup_audio=True,
                output_stem=output_stem,
            )

            message = "哔哩哔哩视频处理完成，下载的视频已删除。"

        else:
            raise RuntimeError("未知的输入方式。")

        download_name = srt_path.name

        return render_template(
            "result.html",
            ok=True,
            message=message,
            srt_name=download_name,
            srt_path=str(srt_path),
        )

    except Exception as e:
        return render_template(
            "result.html",
            ok=False,
            message=str(e),
            srt_name=None,
            srt_path=None,
        ), 500

    finally:
        safe_delete_path(job_audio_dir)

        if source_type == "bilibili":
            safe_delete_path(job_download_dir)

        if source_type == "upload":
            safe_delete_path(job_upload_dir)


@app.route("/download/<filename>", methods=["GET"])
def download_srt(filename):
    filename = secure_filename(filename)
    return send_from_directory(str(SRT_DIR), filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
