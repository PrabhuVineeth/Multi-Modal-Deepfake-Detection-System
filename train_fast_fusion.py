import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from loguru import logger

from config import get_device, model_config
from models.cross_attention import CrossModalFusion
from models.forensic_analyzers import ForensicAnalyzerBundle
from models.evidence_aggregation import ForensicEvidenceAggregator
from models.full_model import ForensicOutput

class FastFusionClassifier(nn.Module):
    def __init__(self, config=model_config):
        super().__init__()
        self.fusion = CrossModalFusion(embed_dim=config.fusion_hidden_dim, num_heads=config.num_attention_heads, num_layers=config.num_cross_attention_layers, dropout=config.attention_dropout)
        self.analyzers = ForensicAnalyzerBundle(input_dim=config.fusion_hidden_dim, hidden_dim=config.analyzer_hidden_dim, dropout=config.analyzer_dropout)
        self.aggregator = ForensicEvidenceAggregator(num_channels=4, hidden_dim=config.evidence_dim, fusion_dim=config.fusion_hidden_dim, use_temporal_attention=config.use_temporal_attention)
        
    def forward(self, audio_emb, face_emb, mouth_emb):
        # Add temporal dimension if needed [B, 512] -> [B, 1, 512]
        if audio_emb.ndim == 2:
            audio_emb = audio_emb.unsqueeze(1)
        if face_emb.ndim == 2:
            face_emb = face_emb.unsqueeze(1)
        if mouth_emb.ndim == 2:
            mouth_emb = mouth_emb.unsqueeze(1)
            
        fusion_out = self.fusion(audio_emb, face_emb, mouth_emb)
        fused = fusion_out.fused_features
        analyzers_out = self.analyzers(fusion_out.speech_lip_features, fusion_out.voice_identity_features, fused, fusion_out.av_sync_features)
        
        logits, _, _ = self.aggregator(analyzers_out, fused)
        return logits

def train_head_on_dataset(dataset_name, emb_dir, epochs=30, batch_size=128, lr=1e-3, device="cuda"):
    logger.info(f"Training fast fusion classifier on {dataset_name} embeddings...")
    data_path = Path(emb_dir) / dataset_name
    
    audio_emb = torch.from_numpy(np.load(data_path / "audio_emb.npy")).float()
    face_emb = torch.from_numpy(np.load(data_path / "face_emb.npy")).float()
    mouth_emb = torch.from_numpy(np.load(data_path / "mouth_emb.npy")).float()
    labels = torch.from_numpy(np.load(data_path / "labels.npy")).float()
    
    dataset = TensorDataset(audio_emb, face_emb, mouth_emb, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = FastFusionClassifier(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(epochs):
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
            logger.info(f"[{dataset_name}] Epoch {epoch+1:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
            
    ckpt_path = Path("checkpoints")
    ckpt_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_path / f"fast_fusion_{dataset_name}.pth")
    logger.info(f"Saved trained head to checkpoints/fast_fusion_{dataset_name}.pth")

def main():
    device = get_device()
    emb_dir = "output/extracted_embeddings"
    for ds in ["fakeavceleb", "faceforensics", "lavdf"]:
        if (Path(emb_dir) / ds / "audio_emb.npy").exists():
            train_head_on_dataset(ds, emb_dir, epochs=30, batch_size=128, device=str(device))

if __name__ == "__main__":
    main()
