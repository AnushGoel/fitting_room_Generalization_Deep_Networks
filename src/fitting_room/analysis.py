"""NumPy-only analysis utilities shared by the dashboard, the report builder, and the tests.

Nothing here imports TensorFlow or SciPy, so the dashboard and the test suite run on a small environment.
Every function mirrors the computation used in the notebook.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

# --------------------------------------------------------------------------------------------
# Forward pass and basic metrics
# --------------------------------------------------------------------------------------------


def forward_full(weights, X):
    """Inference-mode pass through the 784-128-128-10 ReLU network.

    Returns (h1, h2, logits, probs). Dropout is inactive at inference, so it does not appear.
    """
    W1, b1, W2, b2, W3, b3 = weights
    h1 = np.maximum(X @ W1 + b1, 0.0)
    h2 = np.maximum(h1 @ W2 + b2, 0.0)
    z = h2 @ W3 + b3
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return h1, h2, z, e / e.sum(axis=1, keepdims=True)


def forward(weights, X):
    """Class probabilities only."""
    return forward_full(weights, X)[3]


def cross_entropy(p, y):
    """Mean data cross-entropy, with no regularization term."""
    p = np.asarray(p, dtype=np.float64)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))


def accuracy(p, y):
    return float(np.mean(np.asarray(p).argmax(axis=1) == np.asarray(y)))


def brier(p, y, n_classes=10):
    onehot = np.eye(n_classes)[np.asarray(y)]
    return float(np.mean(np.sum((np.asarray(p, dtype=np.float64) - onehot) ** 2, axis=1)))


# --------------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------------


def reliability(probs, y, n_bins=15):
    """Equal-width confidence bins. Returns per-bin accuracy, confidence and counts, plus ECE and MCE."""
    probs = np.asarray(probs, dtype=np.float64)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = pred == np.asarray(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    acc_b, conf_b, cnt_b = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        cnt_b[b] = int(m.sum())
        if cnt_b[b]:
            acc_b[b], conf_b[b] = correct[m].mean(), conf[m].mean()
    gaps = np.abs(acc_b - conf_b)
    return {"edges": edges, "acc": acc_b, "conf": conf_b, "count": cnt_b,
            "ece": float(np.sum(cnt_b / len(conf) * gaps)), "mce": float(gaps[cnt_b > 0].max()) if (cnt_b > 0).any() else 0.0,
            "mean_conf": float(conf.mean()), "accuracy": float(correct.mean())}


# --------------------------------------------------------------------------------------------
# Uncertainty and significance
# --------------------------------------------------------------------------------------------


def wilson(p, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(correct, n_boot=2000, seed=42, level=0.95):
    """Percentile bootstrap interval for the mean of a 0/1 vector."""
    correct = np.asarray(correct, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(correct)
    boots = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    a = (1 - level) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def sampling_band(p, n=10_000, z=1.96):
    """Half-width of an approximate interval for the difference of two independent accuracies near p."""
    return z * math.sqrt(2 * p * (1 - p) / n)


def binom_two_sided_p(k, n):
    """Exact two-sided binomial test of k successes in n trials against p = 0.5 (symmetric case)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2 ** n)))


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar test on paired correctness vectors. Returns (b, c, p): b = A right & B wrong, c = the reverse."""
    a, bb = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    b = int(np.sum(a & ~bb))
    c = int(np.sum(~a & bb))
    return b, c, binom_two_sided_p(min(b, c), b + c)


def holm(pvals):
    """Holm step-down adjustment; returns adjusted p-values in the original order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(np.argsort(pvals)):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# --------------------------------------------------------------------------------------------
# Early stopping and signal propagation
# --------------------------------------------------------------------------------------------


def simulate_early_stopping(values, patience, mode="min"):
    """Replays keras.callbacks.EarlyStopping(min_delta=0, restore_best_weights=True).

    Returns (stop_epoch, restored_epoch), both 1-indexed.
    """
    best, best_ep, wait = (math.inf if mode == "min" else -math.inf), 0, 0
    for e, v in enumerate(values, start=1):
        if (v < best) if mode == "min" else (v > best):
            best, best_ep, wait = v, e, 0
        else:
            wait += 1
            if wait >= patience:
                return e, best_ep
    return len(values), best_ep


def predicted_rms(input_m2, fans, scheme):
    """Predicted RMS of each hidden ReLU layer at initialization (zero biases, zero-mean symmetric weights).

    input_m2 : E[x^2] of the inputs; fans : list of (fan_in, fan_out); scheme : "glorot" or "he".
    Uses E[z^2] = fan_in * Var(W) * E[h^2] and E[relu(z)^2] = E[z^2] / 2 (He et al., 2015).
    """
    m2, out = float(input_m2), []
    for fan_in, fan_out in fans:
        var = 2.0 / (fan_in + fan_out) if scheme == "glorot" else 2.0 / fan_in
        m2 = fan_in * var * m2 / 2.0
        out.append(math.sqrt(m2))
    return out


def relative_signal(width, depth, fan0=784):
    """Activation scale under Glorot relative to He after each of `depth` hidden layers of equal `width`."""
    rel, out = 1.0, [1.0]
    for layer in range(depth):
        fi = fan0 if layer == 0 else width
        glorot_gain = fi * (2.0 / (fi + width)) / 2.0
        he_gain = fi * (2.0 / fi) / 2.0
        rel *= glorot_gain / he_gain
        out.append(math.sqrt(rel))
    return out


# --------------------------------------------------------------------------------------------
# Input perturbations (used by the robustness lab)
# --------------------------------------------------------------------------------------------


def shift_img(img, dx, dy):
    """Shift a 28x28 image right by dx and down by dy, filling with zeros (no wrap-around)."""
    out = np.zeros_like(img)
    ys, yd = slice(max(dy, 0), 28 + min(dy, 0)), slice(max(-dy, 0), 28 + min(-dy, 0))
    xs, xd = slice(max(dx, 0), 28 + min(dx, 0)), slice(max(-dx, 0), 28 + min(-dx, 0))
    out[ys, xs] = img[yd, xd]
    return out


def perturb(vec, noise=0.0, occ=0, occ_x=14, occ_y=14, dx=0, dy=0, contrast=1.0, seed=0):
    """Apply brightness scaling, a shift, a black occluding square, and Gaussian noise, in that order."""
    img = np.asarray(vec, dtype=np.float32).reshape(28, 28).copy()
    if contrast != 1.0:
        img = np.clip(img * contrast, 0, 1)
    if dx or dy:
        img = shift_img(img, int(dx), int(dy))
    if occ > 0:
        y0, x0 = max(0, occ_y - occ // 2), max(0, occ_x - occ // 2)
        img[y0:y0 + occ, x0:x0 + occ] = 0.0
    if noise > 0:
        img = np.clip(img + noise * np.random.default_rng(seed).standard_normal((28, 28)).astype(np.float32), 0, 1)
    return img.ravel()
