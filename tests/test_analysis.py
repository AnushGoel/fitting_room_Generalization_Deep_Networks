import math

import numpy as np

from fitting_room import analysis as A


def test_forward_is_a_probability_distribution():
    rng = np.random.default_rng(0)
    w = [rng.normal(0, 0.05, s).astype(np.float32) for s in [(784, 128), (128,), (128, 128), (128,), (128, 10), (10,)]]
    p = A.forward(w, rng.uniform(0, 1, (32, 784)).astype(np.float32))
    assert p.shape == (32, 10) and np.allclose(p.sum(1), 1, atol=1e-5) and (p >= 0).all()


def test_cross_entropy_limits():
    y = np.arange(10)
    assert A.cross_entropy(np.eye(10), y) < 1e-9
    assert math.isclose(A.cross_entropy(np.full((10, 10), 0.1), y), math.log(10), rel_tol=1e-9)


def test_reliability_detects_overconfidence():
    rng = np.random.default_rng(1)
    n, y = 5000, rng.integers(0, 10, 5000)
    conf = np.full(n, 0.9)
    pred = np.where(rng.uniform(size=n) < 0.9, y, (y + 1) % 10)       # right 90% of the time at 90% confidence
    p = np.full((n, 10), 0.1 / 9); p[np.arange(n), pred] = conf
    calibrated = A.reliability(p, y)
    pred_bad = np.where(rng.uniform(size=n) < 0.6, y, (y + 1) % 10)   # right 60% of the time at 90% confidence
    p_bad = np.full((n, 10), 0.1 / 9); p_bad[np.arange(n), pred_bad] = conf
    over = A.reliability(p_bad, y)
    assert calibrated["count"].sum() == n
    assert calibrated["ece"] < 0.02 and over["ece"] > 0.25


def test_wilson_interval_properties():
    lo, hi = A.wilson(0.9, 10_000)
    assert lo < 0.9 < hi and 0 <= lo and hi <= 1
    lo2, hi2 = A.wilson(0.9, 100)
    assert hi2 - lo2 > hi - lo
    assert math.isclose(hi - lo, 2 * 1.96 * math.sqrt(0.9 * 0.1 / 10_000), rel_tol=0.01)


def test_bootstrap_matches_normal_approximation():
    correct = np.r_[np.ones(900), np.zeros(100)]
    lo, hi = A.bootstrap_ci(correct, n_boot=3000, seed=0)
    assert abs((hi - lo) - 2 * 1.96 * math.sqrt(0.09 / 1000)) < 0.008


def test_exact_binomial_and_mcnemar():
    assert math.isclose(A.binom_two_sided_p(0, 10), 2 / 1024)
    assert A.binom_two_sided_p(5, 10) == 1.0
    a = np.array([1, 1, 1, 0, 0, 1], bool)
    assert A.mcnemar_exact(a, a) == (0, 0, 1.0)
    b, c, p = A.mcnemar_exact(np.r_[np.ones(40), np.zeros(5)].astype(bool), np.r_[np.zeros(40), np.ones(5)].astype(bool))
    assert (b, c) == (40, 5) and p < 1e-6
    assert A.mcnemar_exact(a, ~a)[2] == A.mcnemar_exact(~a, a)[2]


def test_holm_is_monotone_and_bounded():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    adj = A.holm(p)
    assert np.all(adj >= p) and np.all(adj <= 1)
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= 0)


def test_early_stopping_replay():
    loss = [1.0, 0.8, 0.7, 0.75, 0.72, 0.71, 0.9]
    assert A.simulate_early_stopping(loss, patience=2, mode="min") == (5, 3)
    assert A.simulate_early_stopping(loss, patience=10, mode="min") == (7, 3)
    acc = [0.5, 0.6, 0.6, 0.59, 0.61]
    assert A.simulate_early_stopping(acc, patience=2, mode="max") == (4, 2)   # ties do not count as improvement


def test_signal_propagation_theory():
    he = A.predicted_rms(1.0, [(784, 128), (128, 128)], "he")
    gl = A.predicted_rms(1.0, [(784, 128), (128, 128)], "glorot")
    assert np.allclose(he, [1.0, 1.0])
    assert math.isclose((gl[1] / he[1]) ** 2, (784 / 912) * 0.5, rel_tol=1e-9)    # ≈ 0.43 second-moment ratio
    rel = A.relative_signal(128, 20)
    assert math.isclose(rel[2] ** 2, (784 / 912) * 0.5, rel_tol=1e-9) and rel[-1] < 1e-2


def test_perturbations():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 784).astype(np.float32)
    assert np.array_equal(A.perturb(x), x)
    img = x.reshape(28, 28)
    s = A.shift_img(img, 3, -2)
    assert np.array_equal(s[:-2, 3:], img[2:, :-3]) and (s[:, :3] == 0).all() and (s[-2:, :] == 0).all()
    occ = A.perturb(x, occ=6, occ_x=10, occ_y=10).reshape(28, 28)
    assert (occ[7:13, 7:13] == 0).all()
    n1, n2 = A.perturb(x, noise=0.3, seed=5), A.perturb(x, noise=0.3, seed=5)
    assert np.array_equal(n1, n2) and n1.min() >= 0 and n1.max() <= 1
