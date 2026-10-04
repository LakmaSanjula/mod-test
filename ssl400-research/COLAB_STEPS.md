# Colab run guide — SSL400 ST-GCN (accuracy setup)

**Runtime:** Runtime → Change runtime type → **GPU → T4** → Save.

## Target accuracy (previous good run)

| Setting | Value |
|---|---|
| `min_samples_per_class` | **10** (~121 classes) |
| Best val acc / F1 | **~64.5% / ~62.0%** |
| Test acc / F1 / Top-5 | **~62.5% / ~58.8% / ~84.7%** |

If you get ~48% val / ~33% F1, you almost certainly trained on **all-class** processed data (`min_samples=1`, ~300+ classes). Rebuild with `min_samples=10`.

---

## A. Upload to Google Drive

```text
MyDrive/SSL400/
  ssl400-research/          ← latest code
  Dataset - MP - CSV/
```

---

## B. Easiest path — notebook

1. Open `colab_train_stgcn.ipynb` in Colab  
2. Set **GPU (T4)**  
3. Run all cells in order  

The notebook now:
- rewrites the accuracy configs
- **deletes** old `data/processed`, `results`, `checkpoints`
- rebuilds with `min_samples_per_class: 10`
- **asserts** class count is ~121 before training
- trains pure → augmented → compare graphs

---

## C. Manual commands (same accuracy setup)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
from pathlib import Path
PROJECT_DIR = Path('/content/drive/MyDrive/SSL400/ssl400-research')
DATASET_DIR = Path('/content/drive/MyDrive/SSL400/Dataset - MP - CSV')
os.chdir(PROJECT_DIR)
```

```bash
!pip install -q -r requirements.txt
!python scripts/clean_for_retrain.py --also-processed --yes
```

Write `configs/data.yaml` with **`min_samples_per_class: 10`** (not 1), then:

```bash
!python scripts/prepare_dataset.py --data-config configs/data.yaml
```

Check before training:

```python
import json
from pathlib import Path
inv = json.loads(Path('data/inspected/dataset_inventory.json').read_text())
print(inv['num_classes'], inv['min_samples_per_class'], inv['split_counts'])
assert inv['min_samples_per_class'] == 10
assert 100 <= inv['num_classes'] <= 150
```

```bash
!python scripts/train_stgcn.py --model-config configs/model_pure.yaml
!python scripts/evaluate_stgcn.py --checkpoint checkpoints/stgcn_pure_seed42_best.pt --split test --model-config configs/model_pure.yaml

!python scripts/train_stgcn.py --model-config configs/model_augmented.yaml
!python scripts/evaluate_stgcn.py --checkpoint checkpoints/stgcn_augmented_seed42_best.pt --split test --model-config configs/model_augmented.yaml

!python scripts/compare_runs.py
```

---

## D. Do NOT use (drops accuracy)

| Setting | Why |
|---|---|
| `min_samples_per_class: 1` | ~383 classes → accuracy collapses |
| `weighted_sampler: true` | Hurt overall % on this set |
| `class_weights: true` | Hurt overall % on this set |
| Reusing old all-class `data/processed` | Same collapse even if yaml says 10 |

---

## E. Quick failure checks

| Symptom | Fix |
|---|---|
| Classes printed ~300+ | Delete `data/processed`, set min_samples=10, prepare again |
| Val acc stuck ~0.15–0.50 with F1 ≪ acc | Wrong class filter / imbalance from all-class data |
| `CUDA: False` | Runtime → GPU T4 |
| OOM | `batch_size: 16` in model yaml |
