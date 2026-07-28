"""
Train + save a DEPLOYABLE late-fusion head for the UI.

Context: every full-model checkpoint on disk is degenerate (undertrained epoch-1/2,
collapsed to one class), and the working general model was deleted. But today's
corrected pipeline produced a late-fusion config that scored 92.6% AUC
(identity-disjoint) on FakeAVCeleb. This trains that same head (audio-only +
visual-only, averaged) on ALL of the FakeAVCeleb raw features and saves the
weights + StandardScaler stats + threshold so the UI can classify for real.

Output: checkpoints/latefusion_head.pt
"""

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import get_device
from ablation_fakeavceleb import setup_determinism, SEED
from evaluate_modality_ablation import AudioOnlyModel, VisualOnlyModel
from train_joint_balanced import FocalLoss

FEAT = "output/crossdataset_eval_scaleup/feat_fakeavceleb.npz"
OUT = "checkpoints/latefusion_head.pt"
EPOCHS = 60


def train_one(model_cls, Xa, Xf, Xm, y, device):
    torch.manual_seed(SEED)
    ds = TensorDataset(
        torch.tensor(Xa, dtype=torch.float32),
        torch.tensor(Xf, dtype=torch.float32),
        torch.tensor(Xm, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = model_cls().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = FocalLoss(alpha=0.5, gamma=2.0)
    model.train()
    for _ in range(EPOCHS):
        for a, f, m, yb in loader:
            opt.zero_grad()
            loss = crit(model(a.to(device), f.to(device), m.to(device)).squeeze(-1), yb.to(device))
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def sigmoid_scores(model, Xa, Xf, Xm, device):
    return torch.sigmoid(model(
        torch.tensor(Xa, dtype=torch.float32, device=device),
        torch.tensor(Xf, dtype=torch.float32, device=device),
        torch.tensor(Xm, dtype=torch.float32, device=device),
    ).squeeze(-1)).cpu().numpy()


def main():
    setup_determinism()
    device = get_device()
    z = np.load(FEAT, allow_pickle=True)
    Xa, Xf, Xm, y = z["audio"], z["face"], z["mouth"], z["labels"]
    logger.info(f"training on {len(y)} FakeAVCeleb samples ({int((y==0).sum())} real / {int((y==1).sum())} fake)")

    sa, sf, sm = StandardScaler().fit(Xa), StandardScaler().fit(Xf), StandardScaler().fit(Xm)
    Xas, Xfs, Xms = sa.transform(Xa), sf.transform(Xf), sm.transform(Xm)

    audio = train_one(AudioOnlyModel, Xas, Xfs, Xms, y, device)
    visual = train_one(VisualOnlyModel, Xas, Xfs, Xms, y, device)

    # Sanity only (train-set, optimistic) -- the real number is the 92.6 CV AUC.
    late = (sigmoid_scores(audio, Xas, Xfs, Xms, device) + sigmoid_scores(visual, Xas, Xfs, Xms, device)) / 2
    logger.info(f"train-set late-fusion AUC (optimistic sanity check): {roc_auc_score(y, late)*100:.1f}")

    torch.save({
        "audio_state": audio.state_dict(),
        "visual_state": visual.state_dict(),
        "scaler_audio": {"mean": sa.mean_.astype(np.float32), "scale": sa.scale_.astype(np.float32)},
        "scaler_face": {"mean": sf.mean_.astype(np.float32), "scale": sf.scale_.astype(np.float32)},
        "scaler_mouth": {"mean": sm.mean_.astype(np.float32), "scale": sm.scale_.astype(np.float32)},
        "threshold": 0.5,
        "source": FEAT,
        "cv_auc": 92.6,
        "note": "audio-only + visual-only heads on raw ViT/wav2vec2 features; UI averages their sigmoids (late fusion).",
    }, OUT)
    logger.info(f"saved deployable head -> {OUT}")


if __name__ == "__main__":
    main()
