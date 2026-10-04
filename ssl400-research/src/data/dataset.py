"""PyTorch Dataset for prepared SSL400 Pose sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import project_root, resolve_path

AugStrength = Literal["normal", "rare"]


def augment_pose_tensor(x: torch.Tensor, strength: AugStrength = "normal") -> torch.Tensor:
    """Mild train-time augmentation that keeps the sign label valid.

    Used transforms (small, label-preserving):
      - slight scale
      - small coordinate noise
      - tiny in-plane rotation
      - short temporal shift
      - signing-speed resample

    `strength="rare"` only slightly raises apply probability for scarce classes.
    Ranges stay mild so majority-class accuracy and curve reliability are preserved.

    Not used: left-right mirror (changes many signs) and joint dropout
    (the previous strong version hurt accuracy).
    """
    out = x.clone()
    _, t, _ = out.shape
    rare = strength == "rare"

    p_scale = 0.85 if rare else 0.7
    p_noise = 0.85 if rare else 0.7
    p_rot = 0.65 if rare else 0.5
    p_shift = 0.55 if rare else 0.4
    p_speed = 0.65 if rare else 0.5
    noise_std = 0.006 if rare else 0.005
    scale_lo, scale_hi = (0.93, 1.07) if rare else (0.95, 1.05)
    angle_max = 0.10 if rare else 0.08  # ~±6° vs ~±5°
    speed_lo, speed_hi = (0.80, 1.20) if rare else (0.85, 1.15)

    if torch.rand(1).item() < p_scale:
        scale = float(torch.empty(1).uniform_(scale_lo, scale_hi).item())
        out = out * scale

    if torch.rand(1).item() < p_noise:
        out = out + torch.randn_like(out) * noise_std

    # Small rotation in the image plane (x, y). z is left unchanged.
    if out.size(0) >= 2 and torch.rand(1).item() < p_rot:
        angle = float(torch.empty(1).uniform_(-angle_max, angle_max).item())
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        x_coord = out[0].clone()
        y_coord = out[1].clone()
        out[0] = cos_a * x_coord - sin_a * y_coord
        out[1] = sin_a * x_coord + cos_a * y_coord

    if t > 4 and torch.rand(1).item() < p_shift:
        shift = int(torch.randint(-2, 3, (1,)).item())
        if shift != 0:
            out = torch.roll(out, shifts=shift, dims=1)

    # Signing-speed change: resample time, then restore length T.
    if t > 8 and torch.rand(1).item() < p_speed:
        rate = float(torch.empty(1).uniform_(speed_lo, speed_hi).item())
        new_t = max(8, int(round(t * rate)))
        src_idx = torch.linspace(0, t - 1, new_t).round().long().clamp(0, t - 1)
        sampled = out[:, src_idx, :]
        back_idx = torch.linspace(0, new_t - 1, t).round().long().clamp(0, new_t - 1)
        out = sampled[:, back_idx, :]

    return out


class SSL400PoseDataset(Dataset):
    """Loads cached `(C, T, V)` tensors from the preparation manifest."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str = "train",
        augment: bool = False,
        rare_class_augment: bool = False,
        rare_class_max_count: int = 12,
    ) -> None:
        processed_dir = resolve_path(processed_dir, project_root())
        manifest_path = processed_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run: python scripts/prepare_dataset.py"
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest: dict[str, list[dict[str, Any]]] = json.load(f)
        if split not in manifest:
            raise KeyError(f"Split '{split}' not in manifest keys: {list(manifest)}")
        self.records = manifest[split]
        self.split = split
        self.augment = bool(augment) and split == "train"
        self.rare_class_augment = bool(rare_class_augment) and self.augment
        self.rare_class_max_count = int(rare_class_max_count)

        class_map_path = processed_dir / "class_to_idx.json"
        with class_map_path.open("r", encoding="utf-8") as f:
            self.class_to_idx = json.load(f)
        self.num_classes = len(self.class_to_idx)

        counts = self.label_counts()
        self.rare_labels = {
            lab for lab, n in counts.items() if n <= self.rare_class_max_count
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        tensor = np.load(record["tensor_path"]).astype(np.float32)
        x = torch.from_numpy(tensor)  # (C, T, V)
        y = int(record["label"])
        if self.augment:
            strength: AugStrength = (
                "rare"
                if self.rare_class_augment and y in self.rare_labels
                else "normal"
            )
            x = augment_pose_tensor(x, strength=strength)
        return {
            "x": x,
            "y": torch.tensor(y, dtype=torch.long),
            "sample_id": record["sample_id"],
            "class_name": record["class_name"],
        }

    def label_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for r in self.records:
            label = int(r["label"])
            counts[label] = counts.get(label, 0) + 1
        return counts

    def sample_weights(self) -> list[float]:
        """Inverse-frequency weights for WeightedRandomSampler."""
        counts = self.label_counts()
        return [1.0 / float(counts[int(r["label"])]) for r in self.records]


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    return {
        "x": x,
        "y": y,
        "sample_id": [b["sample_id"] for b in batch],
        "class_name": [b["class_name"] for b in batch],
    }
