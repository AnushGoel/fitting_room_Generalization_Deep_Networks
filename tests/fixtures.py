"""Builds a tiny but schema-complete artifacts folder, so the pipeline can be tested without TensorFlow."""
import json

import numpy as np
import pandas as pd

from fitting_room.analysis import forward, reliability

KEYS = ["baseline", "he_init", "l2_0.001", "l2_0.01", "dropout_0.2", "dropout_0.4", "tuned"]
CFG = {"baseline": (None, 0.0, 0.0), "he_init": ("he_normal", 0.0, 0.0), "l2_0.001": ("he_normal", 1e-3, 0.0),
       "l2_0.01": ("he_normal", 1e-2, 0.0), "dropout_0.2": ("he_normal", 0.0, 0.2), "dropout_0.4": ("he_normal", 0.0, 0.4),
       "tuned": ("he_normal", 1e-4, 0.25)}
# (final train acc, peak val acc) chosen to look like a typical run of this study
SHAPE = {"baseline": (0.97, 0.893), "he_init": (0.97, 0.894), "l2_0.001": (0.915, 0.889), "l2_0.01": (0.86, 0.853),
         "dropout_0.2": (0.93, 0.899), "dropout_0.4": (0.895, 0.893), "tuned": (0.935, 0.901)}
NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def _weights(rng, scale=0.05):
    return [rng.normal(0, scale, (784, 128)).astype(np.float32), np.zeros(128, np.float32),
            rng.normal(0, scale, (128, 128)).astype(np.float32), np.zeros(128, np.float32),
            rng.normal(0, scale, (128, 10)).astype(np.float32), np.zeros(10, np.float32)]


def _run(k, E, rng):
    tr_end, va_peak = SHAPE[k]
    t = np.arange(1, E + 1)
    clean = 0.80 + (tr_end - 0.80) * (1 - np.exp(-t / 4))
    tr = clean - CFG[k][2] * 0.08
    va = 0.80 + (va_peak - 0.80) * (1 - np.exp(-t / 2.5)) - (0.004 if CFG[k][1] + CFG[k][2] == 0 else 0) * np.maximum(t - 4, 0) / E
    if k == "dropout_0.4":
        tr = np.minimum(tr, va - 0.003)
    ce = 0.32 + (0.15 if CFG[k][1] + CFG[k][2] == 0 else 0.01) * np.maximum(t - 3, 0) / E
    diag = {"clean_train_acc": [0.1] + list(clean), "clean_train_ce": [2.35] + list(0.5 - 0.3 * clean / clean.max()),
            "val_acc": [0.1] + list(va), "val_ce": [2.36] + list(ce),
            "sqnorm_h1": list(250 + 40 * np.arange(E + 1) * (1 - 50 * CFG[k][1])), "sqnorm_h2": [256.0] * (E + 1),
            "sqnorm_out": [10.0] * (E + 1), "l2_penalty": [CFG[k][1] * 506.0] * (E + 1),
            "dead_h1": [0.0] * (E + 1), "dead_h2": [0.0] * (E + 1), "rms_h1": [0.5] * (E + 1), "rms_h2": [0.5] * (E + 1)}
    b = int(np.argmax(va)) + 1
    vce = np.array(ce)
    s = {"best_epoch": b, "best_val_acc": float(va[b - 1]), "train_acc_at_best": float(tr[b - 1]),
         "clean_train_acc_at_best": float(clean[b - 1]), "gap_at_best": float(tr[b - 1] - va[b - 1]),
         "clean_gap_at_best": float(clean[b - 1] - va[b - 1]), "final_train_acc": float(tr[-1]), "final_val_acc": float(va[-1]),
         "final_gap": float(tr[-1] - va[-1]), "final_clean_gap": float(clean[-1] - va[-1]), "val_ce_min": float(vce.min()),
         "val_ce_min_epoch": int(np.argmin(vce)) + 1, "val_ce_final": float(vce[-1]), "val_ce_rise": float(vce[-1] / vce.min() - 1),
         "val_ce_at_best": float(vce[b - 1]), "init_ce": 2.35, "ep1_train_loss": 0.55, "ep1_train_acc": float(tr[0]),
         "ep1_val_acc": float(va[0]) + (0.004 if k == "he_init" else 0), "mean_val_acc_ep1_5": float(va[:5].mean()) + (0.004 if k == "he_init" else 0),
         "epochs_to_85": 2, "epochs_to_88": 4, "val_jitter": 0.003, "late_val_sd": 0.002, "n_train_below_val": int(np.sum(tr < va)),
         "train_time_s": 60.0, "sec_per_epoch": 1.5}
    hist = {"accuracy": list(tr), "loss": list(0.5 - 0.3 * tr / tr.max() + CFG[k][1] * 50), "val_accuracy": list(va),
            "val_loss": list(ce + CFG[k][1] * 50)}
    init, l2, do = CFG[k]
    return {"label": k, "short": k, "color": "#888888", "config": {"init": init, "l2": l2, "dropout": do}, "seed": 42,
            "history": hist, "diag": diag, "summary": s, "n_params": 118282}


def make_artifacts(path, n_val=60, n_test=80, epochs=8, seeds=(42, 7, 2024)):
    rng = np.random.default_rng(0)
    path.mkdir(parents=True, exist_ok=True)
    runs = {k: _run(k, epochs, rng) for k in KEYS}
    (path / "runs.json").write_text(json.dumps(runs))
    W = {k: _weights(np.random.default_rng(i)) for i, k in enumerate(KEYS)}
    np.savez_compressed(path / "weights.npz", **{f"{k}__{i}": w for k in KEYS for i, w in enumerate(W[k])})
    Xv = rng.integers(0, 256, (n_val, 784), dtype=np.uint8); yv = rng.integers(0, 10, n_val)
    Xt = rng.integers(0, 256, (n_test, 784), dtype=np.uint8); yt = rng.integers(0, 10, n_test)
    tp = forward(W["dropout_0.2"], Xt / 255.0).astype(np.float32)
    np.savez_compressed(path / "predictions.npz", X_val_u8=Xv, y_val=yv, X_test_u8=Xt, y_test=yt, test_probs_final=tp,
                        **{f"val_probs__{k}": forward(W[k], Xv / 255.0).astype(np.float16) for k in KEYS})
    pix = rng.uniform(0, 0.4, 784); pix[:100] = 0.01
    np.savez_compressed(path / "data_stats.npz", mean_imgs=rng.uniform(0, 1, (10, 784)), pixel_std=pix,
                        class_sim=np.eye(10) * 0.8 + 0.1, counts=np.full((10, 3), 100),
                        **{f"w1map__{k}": (pix * (3 if k.startswith("l2") else 1) + rng.uniform(0, 0.05, 784)) for k in KEYS})
    rel = reliability(tp, yt)
    cm = np.zeros((10, 10), int)
    np.add.at(cm, (yt, tp.argmax(1)), 1)
    final = {"selected": "dropout_0.2", "label": "He + dropout (p = 0.2)", "config": {"init": "he_normal", "l2": 0.0, "dropout": 0.2},
             "config_text": "test", "candidates": ["he_init", "l2_0.001", "dropout_0.2"], "best_l2": "l2_0.001", "best_dropout": "dropout_0.2",
             "best_val_acc": runs["dropout_0.2"]["summary"]["best_val_acc"], "best_epoch": runs["dropout_0.2"]["summary"]["best_epoch"],
             "test_acc": runs["dropout_0.2"]["summary"]["best_val_acc"] - 0.004, "test_loss": 0.30, "test_ce": 0.30,
             "ci_bootstrap": [0.889, 0.901], "ci_wilson": [0.889, 0.901], "confusion": cm.tolist(),
             "per_class": [{"class": n, "Precision": 0.9, "Recall": 0.9, "F1": 0.9, "Support": 1000} for n in NAMES],
             "top_confusions": [{"count": 100, "a": "Shirt", "b": "T-shirt/top"}], "spearman_sim_confusion": [0.62, 1e-5],
             "test_reliability": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rel.items()},
             "val_reliability": {}, "class_names": NAMES, "epochs": epochs, "batch_size": 128, "seed": 42, "split_fingerprint": "abc"}
    (path / "final_report.json").write_text(json.dumps(final))
    init = {k: {"rms_measured": [0.4, 0.55, 0.48 if k == "baseline" else 0.70], "rms_predicted": [0.4, 0.52, 0.45 if k == "baseline" else 0.68],
                "hist": {"W1": {"counts": [1, 2, 1], "edges": [-0.1, 0, 0.1, 0.2]}, "W2": {"counts": [1, 2, 1], "edges": [-0.1, 0, 0.1, 0.2]}}}
            for k in ("baseline", "he_init")}
    (path / "init_stats.json").write_text(json.dumps(init))
    (path / "optuna_importance.json").write_text(json.dumps({"method": "fANOVA", "importance": {"dropout": 0.55, "l2": 0.38, "init": 0.07},
                                                             "best_params": {"init": "he_normal", "l2": 1e-4, "dropout": 0.25}}))
    pd.DataFrame([{"model": k, "weights": w, "accuracy": 0.89, "mean_confidence": 0.9, "ece": e, "nll": 0.3, "brier": 0.16}
                  for k in KEYS[:6] for w, e in (("best", 0.01), ("final", 0.05 if k in ("baseline", "he_init") else 0.012))]
                 ).to_csv(path / "calibration.csv", index=False)
    pd.DataFrame([{"model": k, "seed": sd, "best_val_acc": runs[k]["summary"]["best_val_acc"] + 0.001 * i, "best_epoch": 5,
                   "clean_gap": 0.02, "train_acc": 0.9, "rank": 1} for k in KEYS for i, sd in enumerate(seeds)]).to_csv(path / "seed_robustness.csv", index=False)
    pd.DataFrame([{"Model": k, "Training accuracy": "90.00%", "Validation accuracy": "89.00%", "Key observation": "x"} for k in KEYS[:5]]
                 ).to_csv(path / "table1_summary.csv", index=False)
    pd.DataFrame([{"Compared with": k, "Selected right, other wrong (b)": 30, "Other right, selected wrong (c)": 20, "Accuracy difference": "+0.10 pp",
                   "Exact p": "0.2", "Holm-adjusted p": "0.6"} for k in KEYS[:4]]).to_csv(path / "mcnemar.csv", index=False)
    pd.DataFrame({"model": KEYS, "silhouette": 0.1}).to_csv(path / "silhouette.csv", index=False)
    pd.DataFrame({"model": ["baseline"] * 20 + ["dropout_0.2"] * 20, "x": rng.normal(size=40), "y": rng.normal(size=40),
                  "label": list(range(10)) * 4}).to_csv(path / "tsne.csv", index=False)
    pd.DataFrame([{"model": k, "monitor": m, "patience": p, "stop_epoch": 5, "restored_epoch": 3, "val_acc": 0.88, "val_ce": 0.3,
                   "regret_pp": 0.1, "epochs_saved": 3} for k in KEYS[:6] for m in ("val_loss", "val_accuracy") for p in range(1, 4)]
                 ).to_csv(path / "early_stopping_simulation.csv", index=False)
    pd.DataFrame([{"number": i, "value": 0.89 + 0.001 * i, "state": "PRUNED" if i % 4 == 3 else "COMPLETE", "duration": "0:00:30",
                   "params_dropout": 0.05 * (i % 8), "params_init": ["glorot_uniform", "he_normal", "he_uniform"][i % 3],
                   "params_l2": 10 ** (-6 + 0.4 * i), "user_attrs_best_epoch": 6, "user_attrs_curve": json.dumps([0.85, 0.88, 0.89])}
                  for i in range(10)]).to_csv(path / "optuna_trials.csv", index=False)
    for k in KEYS[:2]:
        d = path / "runs" / k
        d.mkdir(parents=True, exist_ok=True)
        r = runs[k]
        (d / "result.json").write_text(json.dumps({"key": k, "tag": k, "seed": 42, "config": r["config"], "history": r["history"],
                                                   "diag": r["diag"], "best_epoch": r["summary"]["best_epoch"],
                                                   "best_val_acc": r["summary"]["best_val_acc"], "train_time_s": 60.0}))
    (path / "figures").mkdir(exist_ok=True)
    return path
