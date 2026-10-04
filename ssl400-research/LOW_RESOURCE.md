# Low-resource SSL — what is done and what is still required

This project follows [SSL400_Development_Guide.md](../SSL400_Development_Guide.md). Sinhala Sign Language here is a **low-resource** setting: few examples per sign, strong class imbalance, pose-only landmarks, and no signer IDs.

## Already in the training code

| Technique | Why it fits low-resource SSL | Where |
|---|---|---|
| MediaPipe Pose sequences instead of raw video | Much less compute; trainable on Colab | `src/data/prepare.py` |
| Keep classes with at least 10 samples | 1-sample classes cannot be learned and dropped accuracy to ~25% | `configs/data.yaml` |
| Rare-class-safe split | Very small classes do not break the split | `src/data/prepare.py` |
| Mild landmark augmentation | Extra views without collecting new signers | `src/data/dataset.py` |
| Signing-speed resample | Same sign, different speed; also named in the guide's robustness section | `src/data/dataset.py` |
| Focal loss (gamma 1) | Pays more attention to hard/scarce classes without the accuracy collapse of full rebalancing | `src/training/losses.py` |
| Dropout in the ST-GCN | Regularizes a small dataset and enables MC Dropout | `src/models/stgcn.py` |
| Top-5 accuracy | The guide's human check only corrects inside the Top-5 | training logs and plots |
| MC Dropout evaluation | Uncertainty from the **same** model; no second network | `scripts/mc_dropout_eval.py` |

Not used, because they hurt this dataset: training all 383 classes, weighted sampling, joint dropout, and left-right mirroring.

Optional (accuracy-preserving) imbalance help: `configs/model_imbalance_safe.yaml`

| Extra | Why it is safe here |
|---|---|
| Rare-class-targeted mild aug (`rare_class_augment`) | Slightly stronger aug only for classes with ≤12 train samples |
| Tiny label smoothing `0.05` | Softens overconfidence without rebalancing the sampler |
| Same architecture / focal γ=1 / no weighted sampler | Keeps the ~62.5% accuracy baseline settings |

Graph reliability rules stay the same: **no aug on val/test**, best checkpoint by **raw val macro-F1**, curves use **raw epoch metrics** (best epoch marked).

## Still required by the research guide (not in this training package yet)

These are the next phases. They should stay separate from this Colab training run:

1. Softmax confidence and predictive entropy (single forward pass).
2. Temperature scaling on the **validation** set only.
3. Accept / verify policy and Top-5 human correction.
4. Replay memory of verified samples only.
5. Balanced replay and periodic continual learning.
6. Forgetting and ablation comparisons.
7. Webcam path using the **same** 33 pose landmarks.

## Two-run comparison graphs

Train the **same ST-GCN** twice so graphs stay comparable:

| Run | Config | What changes |
|---|---|---|
| Pure ST-GCN | `configs/model_pure.yaml` | No aug, CE loss only |
| Augmented | `configs/model_augmented.yaml` | Mild aug + speed resample + focal γ=1 |

Then:

```powershell
python scripts/compare_runs.py
```

Plots: loss overlay, val acc/F1 overlay, overfit-gap overlay, best-metric bars.

## Colab

Open `colab_train_stgcn.ipynb`, select GPU T4, and run the cells in order. The notebook cleans old results, prepares data if needed, trains **pure then augmented**, evaluates both, builds comparison graphs, runs MC Dropout on the augmented model, and exports `.h5`.
