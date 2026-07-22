import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from loguru import logger

from config import get_device, model_config
from models.audio_encoder import AudioEncoder
from models.video_encoder import VideoEncoder, MouthEncoder
from train import forensic_collate_fn
from datasets import FaceForensicsDataset, FakeAVCelebDataset, LAVDFDataset

@torch.no_grad()
def extract_dataset_embeddings(dataset_name, dataset_class, data_root, output_dir, device, max_samples=None):
    logger.info(f"Extracting embeddings for {dataset_name}...")
    save_path = Path(output_dir) / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    dataset = dataset_class(
        data_root, split="all",
        use_cache=True, cache_dir="output/cache",
        max_samples=max_samples
    )
    
    loader = DataLoader(
        dataset, batch_size=8, shuffle=False,
        collate_fn=forensic_collate_fn, num_workers=0
    )

    audio_enc = AudioEncoder(model_name=model_config.wav2vec2_model_name, projection_dim=model_config.fusion_hidden_dim).to(device).eval()
    video_enc = VideoEncoder(model_name=model_config.vit_model_name, projection_dim=model_config.fusion_hidden_dim).to(device).eval()
    mouth_enc = MouthEncoder(model_name=model_config.vit_model_name, projection_dim=model_config.fusion_hidden_dim).to(device).eval()

    all_audio_emb = []
    all_face_emb = []
    all_mouth_emb = []
    all_labels = []

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue

        audio = batch["audio"].to(device)
        faces = batch["face_frames"].to(device)
        mouths = batch["mouth_rois"].to(device)
        labels = batch["label"].cpu().numpy()

        audio_out = audio_enc(audio)
        face_out = video_enc(faces)
        mouth_out = mouth_enc(mouths)

        all_audio_emb.append(audio_out.mean(dim=1).cpu().numpy() if audio_out.ndim == 3 else audio_out.cpu().numpy())
        all_face_emb.append(face_out.mean(dim=1).cpu().numpy() if face_out.ndim == 3 else face_out.cpu().numpy())
        all_mouth_emb.append(mouth_out.mean(dim=1).cpu().numpy() if mouth_out.ndim == 3 else mouth_out.cpu().numpy())
        all_labels.append(labels)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            logger.info(f"[{dataset_name}] Processed {batch_idx + 1}/{len(loader)} batches")

    np.save(save_path / "audio_emb.npy", np.concatenate(all_audio_emb, axis=0))
    np.save(save_path / "face_emb.npy", np.concatenate(all_face_emb, axis=0))
    np.save(save_path / "mouth_emb.npy", np.concatenate(all_mouth_emb, axis=0))
    np.save(save_path / "labels.npy", np.concatenate(all_labels, axis=0))
    logger.info(f"Saved {dataset_name} embeddings successfully to {save_path}")

def main():
    device = get_device()
    output_dir = "output/extracted_embeddings"

    datasets = {
        "fakeavceleb": (FakeAVCelebDataset, "c:/Users/Nitte/Desktop/NNM24AD071/FakeAVCeleb_v1.2", 150),
        "faceforensics": (FaceForensicsDataset, "c:/Users/Nitte/Desktop/NNM24AD071/FaceForensics++_C23", 150),
        "lavdf": (LAVDFDataset, "c:/Users/Nitte/Desktop/NNM24AD071/LAV-DF", 150),
    }

    for name, (cls, root, max_s) in datasets.items():
        extract_dataset_embeddings(name, cls, root, output_dir, device, max_samples=max_s)

if __name__ == "__main__":
    main()
