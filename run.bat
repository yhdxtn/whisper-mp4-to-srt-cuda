@echo off
chcp 65001 >nul
echo ==============================
echo Whisper MP4 to SRT CUDA
echo ==============================

if not exist .venv (
    echo 未找到 .venv，请先运行 setup_env.bat
    pause
    exit /b 1
)

.venv\Scripts\python src\transcribe.py --input videos --model medium --language Chinese --device cuda --audio-format wav

pause
