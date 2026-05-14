@echo off
chcp 65001 >nul
echo ==============================
echo 创建 Python 虚拟环境
echo ==============================

if not exist .venv (
    python -m venv .venv
)

echo ==============================
echo 升级 pip
echo ==============================
.venv\Scripts\python -m pip install --upgrade pip

echo ==============================
echo 安装 CUDA 版 PyTorch
echo ==============================
.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ==============================
echo 安装 Whisper
echo ==============================
.venv\Scripts\pip install -r requirements.txt

echo ==============================
echo 检查 CUDA
echo ==============================
.venv\Scripts\python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo 环境安装完成。
pause
