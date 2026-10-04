# Colab — GitHub code + Drive dataset

## Path layout

| Item | Path |
|---|---|
| Code (GitHub clone) | `/content/mod-test/ssl400-research` |
| Dataset (Drive) | `/content/drive/MyDrive/SSL400/Dataset - MP - CSV` |
| Checkpoints / results (Drive) | `/content/drive/MyDrive/SSL400/runs/` |

Repo: `https://github.com/LakmaSanjula/mod-test.git`  
Inside the repo, training code is in `ssl400-research/`.

## Before Colab

1. **Push** latest `ssl400-research` to GitHub from your PC (repo must not be empty)  
2. Keep only the CSV dataset on Drive under `MyDrive/SSL400/Dataset - MP - CSV`  
3. Colab: **Runtime → GPU (T4)**  
4. Open `ssl400-research/colab_train_stgcn.ipynb` (from GitHub or upload once)  
5. Run all cells — start from cell 1 after any clone error (**Runtime → Restart session** if you see `getcwd` errors)

### If clone fails with `getcwd` / missing `ssl400-research`

- Restart the Colab runtime, then re-run from cell 1  
- The clone cell now does `os.chdir("/content")` before `git clone`  
- If the error says `ssl400-research not found`, the GitHub repo is empty or outdated — push from PC first
## What the notebook does

1. Mount Drive (dataset + saved runs)  
2. `git clone` / `git pull` model code from GitHub  
3. Point `raw_csv_root` at Drive dataset  
4. Save checkpoints/plots to Drive `SSL400/runs/`  
5. Rebuild with `min_samples_per_class: 10` (~121 classes)  
6. Train pure → augmented → compare  

## If your Drive folder name differs

Edit cell **2** only:

```python
DATASET_DIR = Path("/content/drive/MyDrive/SSL400/Dataset - MP - CSV")
DRIVE_RUNS_DIR = Path("/content/drive/MyDrive/SSL400/runs")
REPO_URL = "https://github.com/LakmaSanjula/mod-test.git"
```

## Private GitHub repo

In the clone cell set `USE_TOKEN = True` and paste a GitHub Personal Access Token when prompted.

## Accuracy target

- Classes ~**121** (`min_samples=10`)  
- Augmented: ~**64% val / ~62% test**  
- If classes ~300+, you are on the wrong filter — stop and rebuild
