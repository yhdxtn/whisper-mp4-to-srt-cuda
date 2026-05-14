# GitHub 同步

首次上传：

```cmd
git init
git branch -M main
git add .
git commit -m "Initial commit"
gh repo create whisper-mp4-to-srt-cuda --public --source=. --remote=origin --push
```

后续更新：

```cmd
git add .
git status
git commit -m "Update project"
git push
```

不上传视频、音频、字幕、模型。
