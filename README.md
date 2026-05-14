# whisper-mp4-to-srt-cuda-web-bilibili

Whisper 字幕生成工具，支持命令行和网页端。

## 功能

- 使用 RTX 3070 + CUDA 加速 Whisper
- MP4 / MKV / AVI / MOV / WEBM 等视频生成 SRT 字幕
- 网页端支持三种输入方式：
  - 本机访问：可以直接填本地视频路径，不需要上传视频
  - 其他设备访问：必须上传视频，不能填服务器本地路径
  - 输入哔哩哔哩地址：自动下载视频，生成字幕后删除下载的视频
- 支持命令行指定任意视频路径
- 默认不上传视频、音频、字幕、模型到 GitHub

## 安装依赖

```cmd
cd /d D:\github\whisper-mp4-to-srt-cuda
python install_deps.py
```

如果你已经安装过 PyTorch 和 Whisper，只补装网页和哔哩哔哩依赖：

```cmd
python -m pip install flask yt-dlp imageio-ffmpeg
```

## 检查 CUDA

```cmd
python check_cuda.py
```

正常结果：

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 3070
```

## 命令行使用

处理指定视频：

```cmd
python transcribe.py --input "D:\视频\测试.mp4" --model medium --language Chinese --device cuda
```

处理文件夹：

```cmd
python transcribe.py --input "D:\视频文件夹" --model medium --language Chinese --device cuda
```

生成字幕在：

```text
subtitles/
```

## 网页端使用

启动网页：

```cmd
python app.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

本机访问时，可以直接填本地视频路径，例如：

```text
D:\视频\测试.mp4
```

如果要让局域网其他设备访问，把 `app.py` 最后一行改成：

```python
app.run(host="0.0.0.0", port=7860, debug=False)
```

其他设备访问时，不能使用服务器本地路径，必须上传视频或输入哔哩哔哩链接。

## 哔哩哔哩链接

网页端选择“哔哩哔哩链接”，输入类似：

```text
https://www.bilibili.com/video/BVxxxxxxxxxx
```

处理逻辑：

```text
下载视频
提取音频
Whisper 识别
输出 SRT
删除下载的视频
删除临时音频
```

如果视频需要登录、会员权限或被风控，yt-dlp 可能下载失败。这种情况可以先手动下载视频，再用本地路径或上传方式处理。

## 推荐模型

```text
small   更快，准确率稍低
medium  推荐，速度和准确率比较平衡
turbo   更快，适合长视频
large   可能显存不够，不推荐 8GB 显存直接跑
```

## GitHub 上传

```cmd
git add .
git status
git commit -m "Add web UI and Bilibili support"
git push
```

`.gitignore` 已经排除：

```text
videos/
audio/
subtitles/
uploads/
downloads/
models/
.ffmpeg_bin/
*.mp4
*.mp3
*.wav
*.srt
*.pt
*.pth
*.bin
*.onnx
```
