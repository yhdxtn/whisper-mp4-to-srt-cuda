# whisper-mp4-to-srt-cuda-python39-no-system-ffmpeg

Python 3.9 兼容版 Whisper 字幕生成工具。

这个版本不强制要求你手动安装系统 FFmpeg。  
脚本会优先使用系统 FFmpeg；如果找不到，会自动使用 `imageio-ffmpeg` 包自带的 FFmpeg。

## 安装依赖

如果你已经安装过 `torch` 和 `openai-whisper`，只需要补一个：

```cmd
python -m pip install imageio-ffmpeg
```

或者直接完整安装：

```cmd
python install_deps.py
```

## 检查 CUDA

```cmd
python check_cuda.py
```

正常结果类似：

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 3070
```

## 处理指定视频

```cmd
python transcribe.py --input "D:\your_video.mp4" --model medium --language Chinese --device cuda
```

## 处理指定文件夹

```cmd
python transcribe.py --input "D:\your_video_folder" --model medium --language Chinese --device cuda
```

## 默认用法

把视频放进 `videos` 文件夹，然后运行：

```cmd
python transcribe.py
```

## 输出位置

默认输出：

```text
audio/        提取出的音频
subtitles/    生成的 srt 字幕
```

也可以指定输出目录：

```cmd
python transcribe.py --input "D:\your_video.mp4" --audio-dir "D:\out_audio" --srt-dir "D:\out_srt"
```
