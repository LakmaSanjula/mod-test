#!/usr/bin/env python
"""Compare Pure ST-GCN vs Augmented ST-GCN runs and write comparison graphs.

Usage (from ssl400-research/):
  python scripts/compare_runs.py
  python scripts/compare_runs.py \\
    --pure-history results/stgcn_pure_seed42_history.json \\
    --aug-history results/stgcn_augmented_seed42_history.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.plots import plot_pure_vs_augmented_comparison
from src.utils.config import project_root, resolve_path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_compact(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.exists():
        return None
    data = _load_json(path)
    return {k: v for k, v in data.items() if k not in ("y_true", "y_pred", "sample_ids")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pure vs augmented ST-GCN graphs")
    parser.add_argument(
        "--pure-history",
        default="results/stgcn_pure_seed42_history.json",
        help="History JSON from pure ST-GCN training",
    )
    parser.add_argument(
        "--aug-history",
        default="results/stgcn_augmented_seed42_history.json",
        help="History JSON from augmented ST-GCN training",
    )
    parser.add_argument(
        "--pure-test-metrics",
        default="results/stgcn_pure_seed42_test_metrics.json",
    )
    parser.add_argument(
        "--aug-test-metrics",
        default="results/stgcn_augmented_seed42_test_metrics.json",
    )
    parser.add_argument("--out-dir", default="results/plots")
    parser.add_argument("--prefix", default="compare_pure_vs_augmented")
    args = parser.parse_args()

    root = project_root()
    pure_hist_path = resolve_path(args.pure_history, root)
    aug_hist_path = resolve_path(args.aug_history, root)
    out_dir = resolve_path(args.out_dir, root)

    missing = [p for p in (pure_hist_path, aug_hist_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing history file(s). Train both runs first:\n"
            "  python scripts/train_stgcn.py --model-config configs/model_pure.yaml\n"
            "  python scripts/train_stgcn.py --model-config configs/model_augmented.yaml\n"
            + "\n".join(f"  missing: {p}" for p in missing)
        )

    pure_history = _load_json(pure_hist_path)
    aug_history = _load_json(aug_hist_path)
    pure_metrics = _metrics_compact(resolve_path(args.pure_test_metrics, root))
    aug_metrics = _metrics_compact(resolve_path(args.aug_test_metrics, root))

    written = plot_pure_vs_augmented_comparison(
        pure_history=pure_history,
        aug_history=aug_history,
        out_dir=out_dir,
        prefix=args.prefix,
        pure_metrics=pure_metrics,
        aug_metrics=aug_metrics,
    )

    summary = {
        "pure_history": str(pure_hist_path),
        "aug_history": str(aug_hist_path),
        "pure_best_val_acc": max(h["val_accuracy"] for h in pure_history),
        "pure_best_val_f1": max(h["val_f1_macro"] for h in pure_history),
        "aug_best_val_acc": max(h["val_accuracy"] for h in aug_history),
        "aug_best_val_f1": max(h["val_f1_macro"] for h in aug_history),
        "pure_test": pure_metrics,
        "aug_test": aug_metrics,
        "plots": written,
    }
    summary_path = resolve_path("results", root) / f"{args.prefix}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Comparison plots:")
    for p in written:
        print(" ", p)
    print("Summary:", summary_path)
    print(
        f"Best val — pure acc={summary['pure_best_val_acc']:.4f} F1={summary['pure_best_val_f1']:.4f} | "
        f"aug acc={summary['aug_best_val_acc']:.4f} F1={summary['aug_best_val_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
