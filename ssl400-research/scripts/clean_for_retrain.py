#!/usr/bin/env python
"""Remove previous training outputs so you can retrain cleanly.

Keeps:
  - source code / configs / notebook
  - data/processed tensors (set --also-processed to delete those too)

Removes:
  - results/ (old graphs + metric JSONs)
  - checkpoints/ (old .pt / .h5)
  - __pycache__ folders
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rm_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"Removed {path}")
    else:
        print(f"Skip (missing): {path}")


def _rm_pycache(root: Path) -> None:
    removed = 0
    for p in root.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    print(f"Removed {removed} __pycache__ folder(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean old ST-GCN train outputs")
    parser.add_argument(
        "--also-processed",
        action="store_true",
        help="Also delete data/processed (forces full prepare again)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not ask for confirmation",
    )
    args = parser.parse_args()

    targets = [ROOT / "results", ROOT / "checkpoints"]
    if args.also_processed:
        targets.append(ROOT / "data" / "processed")

    print("Will remove:")
    for t in targets:
        print(" -", t)
    if not args.yes:
        reply = input("Continue? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Cancelled.")
            return

    for t in targets:
        _rm_tree(t)
    _rm_pycache(ROOT / "src")
    _rm_pycache(ROOT / "scripts")

    (ROOT / "results" / "plots").mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    print("Ready for a clean retrain.")
    print("1) Pure:      python scripts/train_stgcn.py --model-config configs/model_pure.yaml")
    print("2) Augmented: python scripts/train_stgcn.py --model-config configs/model_augmented.yaml")
    print("3) Compare:   python scripts/compare_runs.py")


if __name__ == "__main__":
    main()
