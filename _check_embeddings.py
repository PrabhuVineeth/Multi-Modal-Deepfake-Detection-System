import numpy as np
from pathlib import Path

p = Path("output/extracted_embeddings")
total = 0
for d in sorted(p.iterdir()):
    if d.is_dir():
        face = np.load(d / "face_emb.npy")
        labels = np.load(d / "labels.npy")
        real_count = int((labels == 0).sum())
        fake_count = int((labels == 1).sum())
        total += len(labels)
        print(f"{d.name}: shape={face.shape}, real={real_count}, fake={fake_count}")
print(f"\nTOTAL SAMPLES: {total}")
