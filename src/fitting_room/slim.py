"""Storage-aware export of the artifacts the dashboard needs, for committing or deploying.

The full export is roughly 15–25 MB. The slim bundle keeps every summary file, a subset of validation images,
only the misclassified test images, and float16 weights and probabilities, which typically brings it to 3–5 MB
with no visible change in the dashboard.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

SMALL_FILES = ["runs.json", "final_report.json", "init_stats.json", "optuna_importance.json", "data_stats.npz",
               "early_stopping_simulation.csv", "calibration.csv", "mcnemar.csv", "seed_robustness.csv", "optuna_trials.csv",
               "tsne.csv", "silhouette.csv", "table1_summary.csv", "registry.csv"]


def _size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def slim(src, dst, n_val=2000, budget_mb=None) -> dict:
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)

    with np.load(src / "predictions.npz") as P:
        keep = {"X_val_u8": P["X_val_u8"][:n_val], "y_val": P["y_val"][:n_val], "slim": np.array(True)}
        keep.update({k: P[k][:n_val].astype(np.float16) for k in P.files if k.startswith("val_probs__")})
        probs, y = P["test_probs_final"], P["y_test"]
        wrong = np.where(probs.argmax(1) != y)[0]
        keep.update({"X_test_u8": P["X_test_u8"][wrong], "y_test": y[wrong], "test_probs_final": probs[wrong].astype(np.float16),
                     "test_index": wrong.astype(np.int32)})
    np.savez_compressed(dst / "predictions.npz", **keep)

    with np.load(src / "weights.npz") as W:
        np.savez_compressed(dst / "weights.npz", **{k: W[k].astype(np.float16) for k in W.files})

    before = sum((src / n).stat().st_size for n in SMALL_FILES + ["predictions.npz", "weights.npz"] if (src / n).exists()) / 1e6
    after = _size_mb(dst)
    if budget_mb is not None and after > budget_mb:
        raise SystemExit(f"slim bundle is {after:.2f} MB, over the {budget_mb} MB budget")
    return {"before_mb": before, "after_mb": after, "n_val": int(min(n_val, len(keep["y_val"]))), "n_test_errors": int(len(wrong))}
