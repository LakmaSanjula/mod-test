# Data representation decision

## Representation
- Source: `/content/drive/MyDrive/SSL400/Dataset - MP - CSV`
- Format: headerless MediaPipe Pose CSV (`T` rows × 33 `[x,y,z,visibility]` cells)
- Training tensor: `(C, T, V)` with C=3, T=64, V=33

## Why this matches the plan
- Supplied MediaPipe CSVs are the only on-disk landmark representation.
- Labels come from folder names (`Category/Class`).
- Pose graph nodes match CSV column order 0..32.

## Filters and split
- `min_samples_per_class`: 10
- Kept classes: 121
- Split seed: 42
- Ratios: train=0.7, val=0.15, test=0.15
- Split counts: {'train': 2139, 'val': 451, 'test': 451}

## Limitations
- Pose-only (no hands/face) limits fine-grained SSL discrimination.
- No signer IDs → cannot claim unseen-signer evaluation from this split.
- Variable native frame lengths (min=10, max=256, median=63.0) resized to fixed T.
