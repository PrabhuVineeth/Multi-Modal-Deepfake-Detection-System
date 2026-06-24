"""
Offline batch preprocessing utility.
Extracts frames, tracks faces (with RetinaFace), crops mouth ROIs,
and synchronizes audio for all samples, saving them as compressed uint8 `.pt` files.
Includes support for manifest registration, integrity validation, chunked runs, and statistics.
"""

import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
from loguru import logger
from tqdm import tqdm

from config import path_config, preprocess_config
from datasets import FaceForensicsDataset, FakeAVCelebDataset, LAVDFDataset
from datasets.base_dataset import BaseDeepfakeDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Batch Preprocessing and Cache Management")
    parser.add_argument("--fakeavceleb-root", default=str(path_config.fakeavceleb_root),
                        help="Path to FakeAVCeleb dataset root")
    parser.add_argument("--faceforensics-root", default=str(path_config.faceforensics_root),
                        help="Path to FaceForensics++ dataset root")
    parser.add_argument("--lavdf-root", default=str(path_config.lavdf_root),
                        help="Path to LAV-DF dataset root")
    parser.add_argument("--cache-dir", default="output/cache",
                        help="Target directory to save preprocessed .pt files")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to process per dataset (for testing)")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Save manifest and update stats every N samples")
    
    # Cache management & validation actions
    parser.add_argument("--validate", action="store_true", default=False,
                        help="Validate the integrity of the existing cache and exit")
    parser.add_argument("--check-checksums", action="store_true", default=False,
                        help="Perform deep validation calculating SHA256 checksums of cache files")
    parser.add_argument("--no-skip", dest="skip_existing", action="store_false", default=True,
                        help="Force reprocessing of already cached files")
    return parser.parse_args()


def process_dataset(dataset: BaseDeepfakeDataset, cache_dir: Path, skip_existing: bool, chunk_size: int):
    logger.info(f"Starting batch preprocessing for {dataset.__class__.__name__} ({len(dataset)} samples)...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset.cache_dir = cache_dir

    # Deduplicate samples by video path to avoid redundant preprocessing
    unique_samples = {}
    for sample in dataset.samples:
        unique_samples[sample.video_path] = sample
        
    logger.info(f"Deduplicated to {len(unique_samples)} unique video samples.")
    
    # Load manifest
    manifest = dataset._load_manifest()
    if "samples" not in manifest:
        manifest["samples"] = {}
    if "dataset_statistics" not in manifest:
        manifest["dataset_statistics"] = {}
        
    success_count = 0
    skipped_count = 0
    failed_count = 0
    unsaved_manifest_changes = False
    
    start_time = time.time()
    
    pbar = tqdm(unique_samples.items(), desc=f"Preprocessing {dataset.__class__.__name__}")
    for idx, (video_path, sample) in enumerate(pbar):
        key = dataset._get_cache_key(video_path)
        cache_path = cache_dir / f"{key}.pt"
        
        # Resumption / Skip logic
        if skip_existing and key in manifest["samples"] and cache_path.exists():
            # Quick check if sizes match
            expected_size = manifest["samples"][key].get("file_size", 0)
            if expected_size > 0 and cache_path.stat().st_size == expected_size:
                skipped_count += 1
                pbar.set_postfix(success=success_count, skipped=skipped_count, failed=failed_count)
                continue
        
        try:
            # Process video using the dataset's built-in pipeline
            preprocessed = dataset.pipeline.process(video_path)
            
            # Convert to raw unpadded tensors
            tensors = dataset._preprocessed_to_tensors(preprocessed, sample)
            
            # Save using dataset's optimized cache saving function (handles uint8, PCM16, and defers manifest I/O)
            dataset._save_to_cache(video_path, tensors, update_manifest=False)
            
            # Gather sizes and checksums for the in-memory manifest registration
            file_size = cache_path.stat().st_size
            checksum = dataset._get_sha256_checksum(cache_path)
            
            # Register in manifest dictionary in memory
            manifest["samples"][key] = {
                "video_path": video_path,
                "cache_file": f"{key}.pt",
                "file_size": file_size,
                "sha256_checksum": checksum,
                "preprocessing_version": "1.0",
                "metadata": {
                    "dataset_name": sample.dataset_name,
                    "label": sample.label,
                    "split": sample.split,
                }
            }
            success_count += 1
            unsaved_manifest_changes = True
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            failed_count += 1
            
        pbar.set_postfix(success=success_count, skipped=skipped_count, failed=failed_count)
        
        # Save manifest at chunk boundaries
        if (success_count + failed_count) % chunk_size == 0 and unsaved_manifest_changes:
            logger.info("Saving chunk manifest boundary update...")
            manifest["dataset_statistics"]["total_samples"] = len(manifest["samples"])
            manifest["dataset_statistics"]["total_size_bytes"] = sum(
                info.get("file_size", 0) for info in manifest["samples"].values()
            )
            manifest["dataset_statistics"]["last_updated"] = datetime.datetime.utcnow().isoformat() + "Z"
            dataset._save_manifest(manifest)
            unsaved_manifest_changes = False

    # Save final manifest updates
    if unsaved_manifest_changes:
        logger.info("Saving final manifest updates...")
        manifest["dataset_statistics"]["total_samples"] = len(manifest["samples"])
        manifest["dataset_statistics"]["total_size_bytes"] = sum(
            info.get("file_size", 0) for info in manifest["samples"].values()
        )
        manifest["dataset_statistics"]["last_updated"] = datetime.datetime.utcnow().isoformat() + "Z"
        dataset._save_manifest(manifest)
        
    elapsed = time.time() - start_time
    total_processed = success_count + failed_count
    rate = total_processed / elapsed if elapsed > 0 else 0
    total_size_gb = manifest["dataset_statistics"].get("total_size_bytes", 0) / (1024 ** 3)
    
    logger.info(f"Finished {dataset.__class__.__name__}:")
    logger.info(f"  - Succeeded: {success_count}")
    logger.info(f"  - Skipped: {skipped_count}")
    logger.info(f"  - Failed: {failed_count}")
    logger.info(f"  - Total Elapsed: {elapsed:.1f}s ({rate:.2f} samples/sec)")
    logger.info(f"  - Total Cache Footprint: {total_size_gb:.2f} GB")


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    
    if args.validate:
        logger.info(f"Starting cache integrity validation in {cache_dir}...")
        # Create a dummy base dataset class to run the validation utility
        class ManifestValidator(BaseDeepfakeDataset):
            def _load_samples(self) -> None:
                pass
                
        validator = ManifestValidator(root_dir=args.cache_dir, split="train", use_cache=True, cache_dir=str(cache_dir))
        report = validator.validate_cache(check_checksums=args.check_checksums)
        print("\n" + "=" * 50)
        print("           CACHE HEALTH HEALTH REPORT           ")
        print("=" * 50)
        print(f"  Total Registered Samples : {report['total_manifest_entries']}")
        print(f"  Healthy Cache Files      : {report['healthy']}")
        print(f"  Corrupt Deleted          : {report['corrupt_deleted']}")
        print(f"  Missing Removed          : {report['missing_removed']}")
        print(f"  Fully Healthy Cache      : {report['is_fully_healthy']}")
        print("=" * 50 + "\n")
        return
        
    logger.info("Initializing datasets for preprocessing...")
    datasets_to_preprocess = []
    
    if Path(args.fakeavceleb_root).exists():
        logger.info(f"Loading FakeAVCeleb from {args.fakeavceleb_root}")
        datasets_to_preprocess.append(
            FakeAVCelebDataset(args.fakeavceleb_root, split="train", max_samples=args.max_samples)
        )
    else:
        logger.warning(f"FakeAVCeleb directory not found: {args.fakeavceleb_root}")
        
    if Path(args.faceforensics_root).exists():
        logger.info(f"Loading FaceForensics++ from {args.faceforensics_root}")
        datasets_to_preprocess.append(
            FaceForensicsDataset(args.faceforensics_root, split="train", max_samples=args.max_samples)
        )
    else:
        logger.warning(f"FaceForensics++ directory not found: {args.faceforensics_root}")
        
    if Path(args.lavdf_root).exists():
        logger.info(f"Loading LAV-DF from {args.lavdf_root}")
        datasets_to_preprocess.append(
            LAVDFDataset(args.lavdf_root, split="train", max_samples=args.max_samples)
        )
    else:
        logger.warning(f"LAV-DF directory not found: {args.lavdf_root}")
        
    if not datasets_to_preprocess:
        logger.error("No valid dataset directories found. Exiting.")
        return
        
    for ds in datasets_to_preprocess:
        process_dataset(ds, cache_dir, args.skip_existing, args.chunk_size)
        
    logger.info("Offline batch preprocessing completed.")


if __name__ == "__main__":
    main()
