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
    print("Now check CUDA:")
    print("  python check_cuda.py")
    print("\nRun:")
    print("  python transcribe.py")


if __name__ == "__main__":
    main()
