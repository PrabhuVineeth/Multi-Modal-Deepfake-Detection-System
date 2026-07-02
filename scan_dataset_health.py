"""
Dataset integrity scanner for MDDS.
Scans all video files in FaceForensics++ using OpenCV to identify corrupt/unreadable videos.
Saves bad file paths to output/dataset_health/faceforensics_bad_files.txt.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import cv2
from tqdm import tqdm


def check_video_integrity(video_path: Path) -> bool:
    """
    Check if a video is readable and valid.
    Returns True if valid, False if corrupt or unreadable.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
            
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            cap.release()
            return False
            
        # Check basic properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            return False
            
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Scan dataset video integrity")
    parser.add_argument("--data-root", type=str, default="c:/Users/Nitte/Desktop/NNM24AD071/FaceForensics++_C23",
                        help="Path to dataset root directory")
    parser.add_argument("--output-dir", type=str, default="output/dataset_health",
                        help="Output directory to save bad files log")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of concurrent threads to scan")
    args = parser.parse_args()
    
    root_path = Path(args.data_root)
    if not root_path.exists():
        print(f"Error: Data root {args.data_root} does not exist.")
        return
        
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning for all .mp4 videos in {root_path}...")
    video_paths = list(root_path.rglob("*.mp4"))
    print(f"Found {len(video_paths)} videos. Starting integrity check using {args.num_workers} threads...")
    
    bad_files = []
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit tasks
        futures = {executor.submit(check_video_integrity, path): path for path in video_paths}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            path = futures[future]
            try:
                is_valid = future.result()
                if not is_valid:
                    bad_files.append(path)
            except Exception as e:
                print(f"\nError scanning {path.name}: {e}")
                bad_files.append(path)
                
    bad_log_path = output_path / "faceforensics_bad_files.txt"
    with open(bad_log_path, "w", encoding="utf-8") as f:
        for path in sorted(bad_files):
            f.write(f"{path.resolve().as_posix()}\n")
            
    print("\n" + "="*50)
    print("INTEGRITY SCAN COMPLETED")
    print("="*50)
    print(f"Total Scanned: {len(video_paths)}")
    print(f"Corrupt/Bad:  {len(bad_files)} ({len(bad_files)/max(len(video_paths), 1)*100:.2f}%)")
    print(f"Log saved to:  {bad_log_path.resolve()}")
    print("="*50)


if __name__ == "__main__":
    main()
