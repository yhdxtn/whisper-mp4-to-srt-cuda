import subprocess
import sys


def run(cmd):
    print("\n>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    py = sys.executable

    run([py, "-m", "pip", "install", "--upgrade", "pip"])

    run([
        py, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])

    run([py, "-m", "pip", "install", "-r", "requirements.txt"])

    print("\nInstallation finished.")
    print("Check CUDA:")
    print("  python check_cuda.py")
    print("\nCLI:")
    print("  python transcribe.py --input \"D:\\your_video.mp4\"")
    print("\nWeb:")
    print("  python app.py")


if __name__ == "__main__":
    main()
