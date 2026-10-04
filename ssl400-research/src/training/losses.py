"""Losses for imbalanced, low-resource sign classes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss down-weights easy examples so scarce classes are not ignored.

    gamma=0 is standard cross-entropy. gamma around 1 is a mild low-resource setting.
    label_smoothing stays tiny (e.g. 0.05) so graphs/accuracy stay stable.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()
