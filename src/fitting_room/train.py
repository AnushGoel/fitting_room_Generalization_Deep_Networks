"""Headless training: `python -m fitting_room train`.

Trains the configurations in configs/experiments.toml and writes each run to artifacts/runs/<tag>/ in exactly
the format the notebook caches. Training on a server and then opening the notebook therefore costs nothing: the
notebook finds every run in the cache and only renders the analysis. Each run also records provenance (git
commit, library versions, timestamp) for the run registry.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import time
import tomllib
from pathlib import Path

import numpy as np

from .analysis import accuracy, cross_entropy, forward_full


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def load_config(path):
    cfg = tomllib.loads(Path(path).read_text())
    for exp in cfg["experiments"].values():
        if exp.get("init") in ("default", "", None):
            exp["init"] = None          # TOML has no null; "default" means the Keras default (Glorot uniform)
    return cfg


def prepare_data(seed, n_val, clean_subset):
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.datasets import fashion_mnist

    (X_raw, y_raw), _ = fashion_mnist.load_data()
    X_all = X_raw.reshape(len(X_raw), -1).astype("float32") / 255.0
    y_all = y_raw.astype("int32")
    idx_train, idx_val = train_test_split(np.arange(len(X_all)), test_size=n_val, stratify=y_all, random_state=seed, shuffle=True)
    X_train, y_train, X_val, y_val = X_all[idx_train], y_all[idx_train], X_all[idx_val], y_all[idx_val]
    clean_idx = np.random.default_rng(seed).choice(len(X_train), clean_subset, replace=False)
    return dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_clean=X_train[clean_idx], y_clean=y_train[clean_idx])


def build_model(keras, init=None, l2=0.0, dropout=0.0, seed=42, name="mlp"):
    keras.utils.set_random_seed(seed)

    def hidden_kwargs():
        kw = {}
        if init is not None:
            kw["kernel_initializer"] = init
        if l2 and l2 > 0:
            kw["kernel_regularizer"] = keras.regularizers.l2(l2)
        return kw

    inputs = keras.Input(shape=(784,), name="pixels")
    x = keras.layers.Dense(128, activation="relu", name="hidden1", **hidden_kwargs())(inputs)
    if dropout > 0:
        x = keras.layers.Dropout(dropout, name="dropout1")(x)
    x = keras.layers.Dense(128, activation="relu", name="hidden2", **hidden_kwargs())(x)
    if dropout > 0:
        x = keras.layers.Dropout(dropout, name="dropout2")(x)
    outputs = keras.layers.Dense(10, activation="softmax", name="output")(x)
    model = keras.Model(inputs, outputs, name="".join(ch if ch.isalnum() else "_" for ch in name))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def make_diagnostics(keras, data, l2, probe_n):
    class TrainingDiagnostics(keras.callbacks.Callback):
        KEYS = ["clean_train_acc", "clean_train_ce", "val_acc", "val_ce", "sqnorm_h1", "sqnorm_h2", "sqnorm_out",
                "l2_penalty", "dead_h1", "dead_h2", "rms_h1", "rms_h2"]

        def __init__(self):
            super().__init__()
            self.rec = {k: [] for k in self.KEYS}
            self.epoch_times, self.best_val_acc, self.best_epoch, self.best_weights, self.init_weights = [], -np.inf, 0, None, None

        def _record(self):
            w = [np.asarray(a, dtype=np.float32) for a in self.model.get_weights()]
            p_tr = forward_full(w, data["X_clean"])[3]
            h1, h2, _, p_va = forward_full(w, data["X_val"])
            r = self.rec
            r["clean_train_acc"].append(accuracy(p_tr, data["y_clean"])); r["clean_train_ce"].append(cross_entropy(p_tr, data["y_clean"]))
            r["val_acc"].append(accuracy(p_va, data["y_val"])); r["val_ce"].append(cross_entropy(p_va, data["y_val"]))
            sq = [float(np.sum(w[i] ** 2)) for i in (0, 2, 4)]
            r["sqnorm_h1"].append(sq[0]); r["sqnorm_h2"].append(sq[1]); r["sqnorm_out"].append(sq[2])
            r["l2_penalty"].append(float(l2 or 0.0) * (sq[0] + sq[1]))
            r["dead_h1"].append(float(np.mean(np.all(h1[:probe_n] == 0, axis=0))))
            r["dead_h2"].append(float(np.mean(np.all(h2[:probe_n] == 0, axis=0))))
            r["rms_h1"].append(float(np.sqrt(np.mean(h1[:probe_n] ** 2)))); r["rms_h2"].append(float(np.sqrt(np.mean(h2[:probe_n] ** 2))))
            return w

        def on_train_begin(self, logs=None):
            self.init_weights = [w.copy() for w in self._record()]

        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_times.append(time.perf_counter() - self._t0)
            w = self._record()
            if float(logs["val_accuracy"]) > self.best_val_acc:
                self.best_val_acc, self.best_epoch, self.best_weights = float(logs["val_accuracy"]), epoch + 1, [a.copy() for a in w]

    return TrainingDiagnostics()


def train_one(key, exp, seed, proto, data, run_dir, force=False):
    import tensorflow as tf
    from tensorflow import keras

    tag = key if seed == proto["seed"] else f"{key}__seed{seed}"
    out = Path(run_dir) / tag
    if (out / "result.json").exists() and not force:
        print(f"[cache] {tag}")
        return out
    model = build_model(keras, **exp, seed=seed, name=tag)
    diag = make_diagnostics(keras, data, exp.get("l2", 0.0), proto["probe_n"])
    t0 = time.perf_counter()
    hist = model.fit(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]), epochs=proto["epochs"],
                     batch_size=proto["batch_size"], shuffle=True, verbose=0, callbacks=[diag])
    res = {"key": key, "tag": tag, "seed": seed, "config": exp,
           "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
           "diag": diag.rec, "epoch_times": diag.epoch_times, "best_epoch": int(diag.best_epoch), "best_val_acc": float(diag.best_val_acc),
           "train_time_s": float(time.perf_counter() - t0), "n_params": int(model.count_params()),
           "provenance": {"source": "cli", "git_commit": _git_commit(), "tensorflow": tf.__version__, "python": platform.python_version(),
                          "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "batch_size": proto["batch_size"]}}
    out.mkdir(parents=True, exist_ok=True)
    for part, w in {"init": diag.init_weights, "best": diag.best_weights, "final": model.get_weights()}.items():
        np.savez(out / f"{part}_weights.npz", **{f"w{i}": np.asarray(a) for i, a in enumerate(w)})
    (out / "result.json").write_text(json.dumps(res))
    print(f"[train] {tag}: best val acc {100 * res['best_val_acc']:.2f}% at epoch {res['best_epoch']} ({res['train_time_s']:.0f}s)")
    keras.backend.clear_session()
    return out


def train_from_config(path, only=(), all_seeds=False, force=False):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    cfg = load_config(path)
    proto = cfg["protocol"]
    data = prepare_data(proto["seed"], proto["n_val"], proto["clean_subset"])
    run_dir = Path(proto.get("artifacts", "artifacts")) / "runs"
    seeds = [proto["seed"]] + (list(proto.get("extra_seeds", [])) if all_seeds else [])
    keys = [k for k in cfg["experiments"] if not only or k in only]
    for seed in seeds:
        for k in keys:
            train_one(k, cfg["experiments"][k], seed, proto, data, run_dir, force=force)
    from .artifacts import write_registry
    print("registry:", write_registry(run_dir.parent))
