"""
Utility script to download FaceForensics++ and LAV-DF datasets from Kaggle.

Prerequisites:
  1. Install the Kaggle API: pip install kaggle
  2. Place your kaggle.json API key at:
       Windows:  C:\\Users\\<user>\\.kaggle\\kaggle.json
       Linux:    ~/.kaggle/kaggle.json
     (Get it from https://www.kaggle.com/settings → "Create New Token")

Usage:
  python download_kaggle_datasets.py                   # Download both datasets
  python download_kaggle_datasets.py --dataset ff++     # Only FaceForensics++
  python download_kaggle_datasets.py --dataset lavdf    # Only LAV-DF
  python download_kaggle_datasets.py --output-dir D:\\data  # Custom output directory
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Default Kaggle dataset slugs (user/dataset-name)
# NOTE: These are community-uploaded datasets on Kaggle. Verify the exact
# slug on kaggle.com if downloads fail — slugs may change over time.
KAGGLE_DATASETS = {
    "ff++": {
        "slug": "sophatvathana/faceforensics",
        "description": "FaceForensics++ (FF++) — face manipulation dataset",
        "output_folder": "FaceForensics++",
    },
    "lavdf": {
        "slug": "bibek777/lavdf-localized-audio-visual-deepfake-dataset",
        "description": "LAV-DF — Localized Audio-Visual Deepfake dataset",
        "output_folder": "LAV-DF",
    },
}

# Default output directory: same parent as the project
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent


def check_kaggle_installed() -> bool:
    """Check if the Kaggle CLI is installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "--version"],
            capture_output=True, text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials exist."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    # Also check KAGGLE_USERNAME env var
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def install_kaggle():
    """Install the kaggle package."""
    print("Installing kaggle package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    print("✅ kaggle package installed successfully.\n")


def download_dataset(slug: str, output_dir: Path):
    """Download and unzip a Kaggle dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading to: {output_dir}")
    print(f"  Kaggle slug: {slug}")
    print(f"  This may take a while depending on dataset size...\n")

    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", slug,
        "-p", str(output_dir),
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✅ Download complete!\n")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"  ❌ Download failed!")
        if result.stderr:
            print(f"  Error: {result.stderr}")
        print(
            "\n  Possible issues:\n"
            "    - The dataset slug may have changed on Kaggle.\n"
            "    - You may need to accept the dataset's terms on kaggle.com first.\n"
            "    - Check your internet connection.\n"
            f"    - Try manually: kaggle datasets download -d {slug}\n"
        )
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download FaceForensics++ and LAV-DF datasets from Kaggle"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ff++", "lavdf", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_base = Path(args.output_dir)
    print("=" * 60)
    print("  Kaggle Dataset Downloader")
    print("  Deepfake Forensic Detection System")
    print("=" * 60)
    print()

    # Step 1: Check/install kaggle
    if not check_kaggle_installed():
        print("⚠️  Kaggle CLI not found. Installing...")
        try:
            install_kaggle()
        except Exception as e:
            print(f"❌ Failed to install kaggle: {e}")
            print("   Please run: pip install kaggle")
            sys.exit(1)
    else:
        print("✅ Kaggle CLI is installed.")

    # Step 2: Check credentials
    if not check_kaggle_credentials():
        print(
            "\n❌ Kaggle API credentials not found!\n\n"
            "To set up credentials:\n"
            "  1. Go to https://www.kaggle.com/settings\n"
            "  2. Click 'Create New Token' under the API section\n"
            "  3. Save the downloaded kaggle.json to:\n"
            f"     {Path.home() / '.kaggle' / 'kaggle.json'}\n"
        )
        sys.exit(1)
    else:
        print("✅ Kaggle credentials found.\n")

    # Step 3: Determine which datasets to download
    if args.dataset == "all":
        datasets_to_download = list(KAGGLE_DATASETS.keys())
    else:
        datasets_to_download = [args.dataset]

    # Step 4: Download
    success_count = 0
    for key in datasets_to_download:
        info = KAGGLE_DATASETS[key]
        target_dir = output_base / info["output_folder"]

        print(f"{'─' * 60}")
        print(f"📦 {info['description']}")
        print(f"{'─' * 60}")

        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"  ⚠️  Directory already exists and is not empty: {target_dir}")
            resp = input("  Overwrite? [y/N]: ").strip().lower()
            if resp != "y":
                print("  Skipped.\n")
                continue

        if download_dataset(info["slug"], target_dir):
            success_count += 1

    # Summary
    print("=" * 60)
    print(f"  Done! {success_count}/{len(datasets_to_download)} datasets downloaded.")
    if success_count > 0:
        print(f"  Output directory: {output_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()
