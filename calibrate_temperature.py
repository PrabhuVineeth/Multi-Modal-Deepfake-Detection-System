import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from train_fast_fusion import FastFusionClassifier
from config import get_device, model_config
from utils.metrics import compute_ece, compute_auc

def calibrate_temperature(logits: np.ndarray, labels: np.ndarray):
    """Find scalar temperature T to minimize ECE."""
    best_t = 1.0
    best_ece = 1.0
    
    for t in np.arange(0.1, 5.0, 0.05):
        scaled_logits = logits / t
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        preds = (probs >= 0.5).astype(int)
        confidences = np.where(preds == 1, probs, 1.0 - probs)
        ece = compute_ece(confidences, preds, labels)
        if ece < best_ece:
            best_ece = ece
            best_t = t
            
    return float(best_t), float(best_ece)

def run_calibration():
    device = get_device()
    emb_dir = Path("output/extracted_embeddings")
    
    model = FastFusionClassifier(model_config).to(device)
    model.load_state_dict(torch.load("checkpoints/fast_fusion_joint_balanced.pth", map_location=device))
    model.eval()
    
    all_audio, all_face, all_mouth, all_labels = [], [], [], []
    for ds in ["fakeavceleb", "faceforensics", "lavdf"]:
        p = emb_dir / ds
        if p.exists():
            all_audio.append(np.load(p / "audio_emb.npy"))
            all_face.append(np.load(p / "face_emb.npy"))
            all_mouth.append(np.load(p / "mouth_emb.npy"))
            all_labels.append(np.load(p / "labels.npy"))
            
    audio_t = torch.from_numpy(np.concatenate(all_audio, axis=0)).float().to(device)
    face_t = torch.from_numpy(np.concatenate(all_face, axis=0)).float().to(device)
    mouth_t = torch.from_numpy(np.concatenate(all_mouth, axis=0)).float().to(device)
    labels = np.concatenate(all_labels, axis=0)
    
    with torch.no_grad():
        logits = model(audio_t, face_t, mouth_t).squeeze(-1).cpu().numpy()
        
    raw_probs = 1.0 / (1.0 + np.exp(-logits))
    raw_preds = (raw_probs >= 0.5).astype(int)
    raw_conf = np.where(raw_preds == 1, raw_probs, 1.0 - raw_probs)
    uncalibrated_ece = compute_ece(raw_conf, raw_preds, labels)
    
    best_t, calibrated_ece = calibrate_temperature(logits, labels)
    
    logger.info(f"Uncalibrated ECE: {uncalibrated_ece:.4f}")
    logger.info(f"Optimal Temperature T: {best_t:.2f}")
    logger.info(f"Calibrated ECE: {calibrated_ece:.4f} (-{(uncalibrated_ece - calibrated_ece)/uncalibrated_ece*100:.1f}%)")

if __name__ == "__main__":
    run_calibration()
