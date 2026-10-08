"""Reading the notebook's exported results, and building the run registry from the training cache."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

JSON_FILES = {"runs": "runs.json", "final": "final_report.json", "init": "init_stats.json", "imp": "optuna_importance.json"}
NPZ_FILES = {"preds": "predictions.npz", "stats": "data_stats.npz"}
CSV_FILES = {"es": "early_stopping_simulation.csv", "cal": "calibration.csv", "mc": "mcnemar.csv",
             "seeds": "seed_robustness.csv", "trials": "optuna_trials.csv", "tsne": "tsne.csv",
             "sil": "silhouette.csv", "table1": "table1_summary.csv", "registry": "registry.csv"}


def _npz(path):
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def load(path) -> dict:
    """Load every artifact that exists. Missing files come back as None, so callers can degrade gracefully."""
    p = Path(path)
    A = {"path": p}
    for key, name in JSON_FILES.items():
        f = p / name
        A[key] = json.loads(f.read_text()) if f.exists() else None
    for key, name in NPZ_FILES.items():
        A[key] = _npz(p / name)
    for key, name in CSV_FILES.items():
        f = p / name
        A[key] = pd.read_csv(f) if f.exists() else None
    if A["registry"] is None and (p / "runs").exists():
        rows = registry_rows(p / "runs")
        A["registry"] = pd.DataFrame(rows) if rows else None

    weights = {}
    for name, arr in (_npz(p / "weights.npz") or {}).items():
        key, idx = name.rsplit("__", 1)
        weights.setdefault(key, {})[int(idx)] = arr.astype(np.float32)
    A["weights"] = {k: [v[i] for i in range(len(v))] for k, v in weights.items()}

    if A["preds"] is not None:
        A["X_val"] = A["preds"]["X_val_u8"].astype(np.float32) / 255.0
        A["X_test"] = A["preds"]["X_test_u8"].astype(np.float32) / 255.0
        A["slim"] = bool(A["preds"].get("slim", np.array(False)))
    else:
        A["slim"] = False
    return A


def config_hash(config: dict, epochs: int, batch_size: int | None = None) -> str:
    """Short, stable fingerprint of everything that defines a training run except the seed."""
    blob = json.dumps({"config": config, "epochs": epochs, "batch_size": batch_size}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def registry_rows(runs_dir) -> list[dict]:
    """One row per cached run (from the notebook or the CLI), newest information first."""
    rows = []
    for f in sorted(Path(runs_dir).glob("*/result.json")):
        r = json.loads(f.read_text())
        h, d = r["history"], r["diag"]
        b = int(r["best_epoch"])
        prov = r.get("provenance", {})
        rows.append({
            "tag": r["tag"], "model": r["key"], "seed": r["seed"],
            "config_hash": config_hash(r["config"], len(h["val_accuracy"]), prov.get("batch_size")),
            "init": r["config"].get("init") or "glorot_uniform (default)", "l2": r["config"].get("l2", 0.0),
            "dropout": r["config"].get("dropout", 0.0), "epochs_run": len(h["val_accuracy"]),
            "best_epoch": b, "best_val_acc": round(float(r["best_val_acc"]), 5),
            "clean_train_acc_at_best": round(float(d["clean_train_acc"][b]), 5),
            "clean_gap_at_best": round(float(d["clean_train_acc"][b] - h["val_accuracy"][b - 1]), 5),
            "min_val_ce": round(float(min(d["val_ce"][1:])), 5), "train_time_s": round(float(r["train_time_s"]), 1),
            "source": prov.get("source", "notebook"), "git_commit": prov.get("git_commit", ""),
            "tensorflow": prov.get("tensorflow", ""), "created_utc": prov.get("created_utc", ""),
        })
    return rows


def write_registry(art_dir) -> Path:
    art_dir = Path(art_dir)
    rows = registry_rows(art_dir / "runs")
    out = art_dir / "registry.csv"
    if rows:
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return out
