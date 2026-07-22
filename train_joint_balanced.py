import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from loguru import logger

from config import get_device, model_config
from train_fast_fusion import FastFusionClassifier

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def train_joint_balanced():
    device = get_device()
    emb_dir = Path("output/extracted_embeddings")

    all_audio, all_face, all_mouth, all_labels = [], [], [], []
    for ds in ["fakeavceleb", "faceforensics", "lavdf"]:
        p = emb_dir / ds
        if p.exists():
            all_audio.append(np.load(p / "audio_emb.npy"))
            all_face.append(np.load(p / "face_emb.npy"))
            all_mouth.append(np.load(p / "mouth_emb.npy"))
            all_labels.append(np.load(p / "labels.npy"))

    audio_cat = np.concatenate(all_audio, axis=0)
    face_cat = np.concatenate(all_face, axis=0)
    mouth_cat = np.concatenate(all_mouth, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)

    # Balance datasets by oversampling minority class (real samples)
    real_idx = np.where(labels_cat == 0)[0]
    fake_idx = np.where(labels_cat == 1)[0]
    logger.info(f"Original Joint dataset: Real={len(real_idx)}, Fake={len(fake_idx)}")

    # Oversample real samples to match fake count
    oversampled_real_idx = np.random.choice(real_idx, size=len(fake_idx), replace=True)
    balanced_idx = np.concatenate([fake_idx, oversampled_real_idx])
    np.random.shuffle(balanced_idx)

    audio_b = torch.from_numpy(audio_cat[balanced_idx]).float()
    face_b = torch.from_numpy(face_cat[balanced_idx]).float()
    mouth_b = torch.from_numpy(mouth_cat[balanced_idx]).float()
    labels_b = torch.from_numpy(labels_cat[balanced_idx]).float()

    logger.info(f"Balanced Joint dataset size: {len(labels_b)} (50% Real / 50% Fake)")

    dataset = TensorDataset(audio_b, face_b, mouth_b, labels_b)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = FastFusionClassifier(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    criterion = FocalLoss(alpha=0.5, gamma=2.0)

    model.train()
    for epoch in range(40):
        total_loss = 0.0
        correct = 0
        total = 0
        for a, f, m, y in loader:
            a, f, m, y = a.to(device), f.to(device), m.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(a, f, m).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)

        acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"[Joint Balanced] Epoch {epoch+1:02d}/40 | Focal Loss: {avg_loss:.4f} | Balanced Acc: {acc:.4f}")

    ckpt_path = Path("checkpoints")
    torch.save(model.state_dict(), ckpt_path / "fast_fusion_joint_balanced.pth")
    logger.info("Saved balanced joint fusion classifier to checkpoints/fast_fusion_joint_balanced.pth")

if __name__ == "__main__":
    train_joint_balanced()
