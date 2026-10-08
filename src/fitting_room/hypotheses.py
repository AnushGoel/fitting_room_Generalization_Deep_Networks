"""Pre-registered hypotheses, each with a fixed decision rule that is applied to the exported results.

The statements and thresholds below were written before the full experiment was run (see docs/HYPOTHESES.md).
A hypothesis is "supported" when its rule passes, "not supported" when it fails, and "inconclusive" when the
artifacts needed to test it are missing (for example, when the seed study or the search was switched off).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .analysis import sampling_band

SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE = "supported", "not supported", "inconclusive"


@dataclass
class Hypothesis:
    id: str
    topic: str
    statement: str
    rule: str
    basis: str
    test: Callable[[dict], tuple]
    if_supported: str
    if_not: str


@dataclass
class Result:
    hypothesis: Hypothesis
    verdict: str
    evidence: str
    values: dict = field(default_factory=dict)

    @property
    def conclusion(self) -> str:
        if self.verdict == INCONCLUSIVE:
            return f"{self.hypothesis.topic}: not tested in this run ({self.evidence})."
        template = self.hypothesis.if_supported if self.verdict == SUPPORTED else self.hypothesis.if_not
        return template.format(**self.values)


def _s(A, k):
    return A["runs"][k]["summary"]


def _pp(x):
    return f"{100 * x:+.2f} pp"


def _pc(x):
    return f"{100 * x:.2f}%"


def _need(A, *keys):
    missing = [k for k in keys if A.get(k) is None]
    if missing:
        raise LookupError("missing " + ", ".join(missing))


# ---------------------------------------------------------------------------------- tests
def t_signal(A):
    _need(A, "init")
    pred = A["init"]["baseline"]["rms_predicted"][2] / A["init"]["he_init"]["rms_predicted"][2]
    meas = A["init"]["baseline"]["rms_measured"][2] / A["init"]["he_init"]["rms_measured"][2]
    rel_err = abs(meas - pred) / pred
    v = dict(pred=f"{pred:.3f}", meas=f"{meas:.3f}", err=f"{100 * rel_err:.1f}%")
    return rel_err <= 0.15, f"predicted ratio {v['pred']}, measured {v['meas']} (error {v['err']})", v


def t_init_early(A):
    b, h = _s(A, "baseline"), _s(A, "he_init")
    early = h["mean_val_acc_ep1_5"] - b["mean_val_acc_ep1_5"]
    late = h["best_val_acc"] - b["best_val_acc"]
    v = dict(early=_pp(early), late=_pp(late))
    return abs(early) > abs(late), f"Δ mean val. acc. epochs 1–5 = {v['early']}; Δ best val. acc. = {v['late']}", v


def t_init_final(A):
    b, h = _s(A, "baseline"), _s(A, "he_init")
    seeds = A.get("seeds")
    if seeds is not None and {"baseline", "he_init"} <= set(seeds.model):
        g = seeds.groupby("model").best_val_acc
        diff = g.mean()["he_init"] - g.mean()["baseline"]
        noise = 2 * max(g.std()["he_init"], g.std()["baseline"])
        n = int(seeds.seed.nunique())
        how = f"{n}-seed means; noise threshold 2 × max SD = {100 * noise:.2f} pp"
    else:
        diff = h["best_val_acc"] - b["best_val_acc"]
        noise = sampling_band(b["best_val_acc"])
        how = f"single seed; sampling band ±{100 * noise:.2f} pp"
    v = dict(diff=_pp(diff), how=how)
    return abs(diff) < noise, f"He − Glorot best validation accuracy {v['diff']} ({how})", v


def t_baseline_conf(A):
    s = _s(A, "baseline")
    drop = s["best_val_acc"] - s["final_val_acc"]
    v = dict(rise=f"{100 * s['val_ce_rise']:.0f}%", m=s["val_ce_min_epoch"], drop=f"{100 * drop:.2f} pp")
    return s["val_ce_rise"] > 0.10 and drop < 0.01, (f"validation CE rises {v['rise']} after its minimum at epoch {v['m']}; "
                                                     f"validation accuracy at the last epoch is {v['drop']} below its best"), v


def t_l2_strong(A):
    h, s = _s(A, "he_init"), _s(A, "l2_0.01")
    dtr = s["clean_train_acc_at_best"] - h["clean_train_acc_at_best"]
    dva = s["best_val_acc"] - h["best_val_acc"]
    band = sampling_band(h["best_val_acc"])
    v = dict(dtr=_pp(dtr), dva=_pp(dva))
    return dtr < -0.02 and dva < -band, f"vs. He: clean training accuracy {v['dtr']}, best validation accuracy {v['dva']}", v


def _gap_cut(A, k):
    h, s = _s(A, "he_init"), _s(A, k)
    if h["clean_gap_at_best"] <= 0:
        raise LookupError("He model has no positive gap to reduce")
    return h, s, 1 - s["clean_gap_at_best"] / h["clean_gap_at_best"]


def t_l2_moderate(A):
    h, s, cut = _gap_cut(A, "l2_0.001")
    v = dict(g0=f"{100 * h['clean_gap_at_best']:.2f} pp", g1=f"{100 * s['clean_gap_at_best']:.2f} pp", cut=f"{100 * cut:.0f}%",
             dva=_pp(s["best_val_acc"] - h["best_val_acc"]))
    return cut >= 0.30, f"clean gap {v['g0']} → {v['g1']} ({v['cut']} smaller); best validation accuracy {v['dva']} vs. He", v


def t_dropout_balance(A):
    h, s, cut = _gap_cut(A, "dropout_0.2")
    dva = s["best_val_acc"] - h["best_val_acc"]
    band = sampling_band(h["best_val_acc"])
    v = dict(g0=f"{100 * h['clean_gap_at_best']:.2f} pp", g1=f"{100 * s['clean_gap_at_best']:.2f} pp", cut=f"{100 * cut:.0f}%",
             dva=_pp(dva), band=f"{100 * band:.2f} pp")
    return cut >= 0.30 and dva > -band, (f"clean gap {v['g0']} → {v['g1']} ({v['cut']} smaller); best validation accuracy "
                                         f"{v['dva']} vs. He (band ±{v['band']})"), v


def t_dropout_artifact(A):
    s = _s(A, "dropout_0.4")
    v = dict(n=s["n_train_below_val"], clean=_pc(s["clean_train_acc_at_best"]), val=_pc(s["best_val_acc"]))
    return s["n_train_below_val"] >= 1 and s["clean_train_acc_at_best"] >= s["best_val_acc"], (
        f"Keras training accuracy below validation in {v['n']} epochs; at the best epoch clean training {v['clean']} "
        f"vs. validation {v['val']}"), v


def t_calibration(A):
    _need(A, "cal")
    sel = A["final"]["selected"]
    reg = sel if sel not in ("baseline", "he_init") else max(
        ["l2_0.001", "l2_0.01", "dropout_0.2", "dropout_0.4"], key=lambda k: _s(A, k)["best_val_acc"])
    cal = A["cal"].set_index(["model", "weights"]).ece
    d_he = cal[("he_init", "final")] - cal[("he_init", "best")]
    d_reg = cal[(reg, "final")] - cal[(reg, "best")]
    v = dict(reg=A["runs"][reg]["label"], d_he=_pp(d_he), d_reg=_pp(d_reg))
    return d_he > d_reg, f"ECE change from best epoch to the last epoch: He {v['d_he']}, {v['reg']} {v['d_reg']}", v


def t_l2_inputs(A):
    _need(A, "stats")
    st = A["stats"]
    r_he = float(np.corrcoef(st["w1map__he_init"], st["pixel_std"])[0, 1])
    r_l2 = float(np.corrcoef(st["w1map__l2_0.001"], st["pixel_std"])[0, 1])
    v = dict(r_he=f"{r_he:.2f}", r_l2=f"{r_l2:.2f}")
    return r_l2 > r_he, f"correlation of first-layer weight magnitude with pixel variability: He {v['r_he']}, L2 0.001 {v['r_l2']}", v


def t_errors(A):
    rho, p = A["final"]["spearman_sim_confusion"]
    v = dict(rho=f"{rho:.2f}", p=f"{p:.1e}")
    return rho > 0.3 and p < 0.05, f"Spearman ρ = {v['rho']} (p = {v['p']}) across 45 class pairs", v


def t_tuning(A):
    _need(A, "imp")
    imp = A["imp"]["importance"]
    v = dict(imp=", ".join(f"{k} {val:.2f}" for k, val in sorted(imp.items(), key=lambda kv: -kv[1])), method=A["imp"]["method"])
    return min(imp, key=imp.get) == "init", f"importance ({v['method']}): {v['imp']}", v


def t_transfer(A):
    F = A["final"]
    d = F["test_acc"] - F["best_val_acc"]
    v = dict(val=_pc(F["best_val_acc"]), test=_pc(F["test_acc"]), d=_pp(d), lo=_pc(F["ci_wilson"][0]), hi=_pc(F["ci_wilson"][1]))
    return abs(d) < 0.01, f"validation {v['val']}, test {v['test']} ({v['d']}); 95% Wilson interval {v['lo']} to {v['hi']}", v


# ---------------------------------------------------------------------------------- registry
HYPOTHESES = [
    Hypothesis("H1", "Signal propagation",
               "At initialization, the ratio of second-hidden-layer activation scale under Glorot vs. He matches the analytical prediction.",
               "Measured ratio within ±15% of the predicted ratio.", "He et al. (2015); Glorot & Bengio (2010)", t_signal,
               "Variance-propagation theory predicts the untrained network well: the Glorot/He activation ratio was {meas} against a predicted {pred} (error {err}).",
               "The variance-propagation prediction missed: measured ratio {meas} against a predicted {pred} (error {err})."),
    Hypothesis("H2", "Initialization acts early",
               "The effect of He vs. Glorot initialization on validation accuracy is larger in epochs 1–5 than at the best epoch.",
               "|Δ mean validation accuracy, epochs 1–5| > |Δ best validation accuracy|.", "Kingma & Ba (2015); He et al. (2015)", t_init_early,
               "Initialization mattered mostly at the start: the early difference was {early}, the best-epoch difference {late}.",
               "Initialization did not act mainly early: the early difference was {early}, the best-epoch difference {late}."),
    Hypothesis("H3", "Initialization and final accuracy",
               "He initialization does not change best validation accuracy beyond run-to-run noise in this two-layer network.",
               "|Δ| below twice the larger seed SD (or below the sampling band with one seed).", "Bouthillier et al. (2021)", t_init_final,
               "He initialization did not change the final result beyond noise ({diff}; {how}).",
               "He initialization changed best validation accuracy beyond noise ({diff}; {how})."),
    Hypothesis("H4", "Baseline overfits in confidence",
               "Without regularization, validation cross-entropy rises after an early minimum while validation accuracy stays near its peak.",
               "Validation CE ends >10% above its minimum and final validation accuracy is <1 pp below its best.", "Guo et al. (2017)", t_baseline_conf,
               "The unregularized baseline overfit in confidence rather than in accuracy: validation CE rose {rise} after epoch {m}, while accuracy ended only {drop} below its best.",
               "The baseline did not show the predicted confidence-only overfitting (CE rise {rise}; accuracy drop {drop})."),
    Hypothesis("H5", "Strong L2 underfits",
               "L2 with λ = 0.01 lowers both clean training accuracy and best validation accuracy relative to He.",
               "Clean training accuracy down by >2 pp and validation accuracy down by more than the sampling band.", "Krogh & Hertz (1991)", t_l2_strong,
               "λ = 0.01 was too strong: relative to He, clean training accuracy changed by {dtr} and best validation accuracy by {dva}, the signature of underfitting.",
               "λ = 0.01 did not underfit as predicted (clean training {dtr}, validation {dva})."),
    Hypothesis("H6", "Moderate L2 narrows the gap",
               "L2 with λ = 0.001 reduces the clean train–validation gap by at least 30% relative to He.",
               "Clean gap at the best epoch at most 70% of He's.", "Krogh & Hertz (1991); Goodfellow et al. (2016)", t_l2_moderate,
               "A moderate penalty (λ = 0.001) narrowed the gap from {g0} to {g1} ({cut} smaller), at a validation cost of {dva}.",
               "λ = 0.001 narrowed the gap less than predicted ({g0} → {g1}, {cut})."),
    Hypothesis("H7", "Dropout balances fit and generalization",
               "Dropout at p = 0.2 reduces the clean gap by at least 30% without lowering best validation accuracy beyond noise.",
               "Clean gap at most 70% of He's and Δ validation accuracy above −(sampling band).", "Srivastava et al. (2014)", t_dropout_balance,
               "Dropout at p = 0.2 cut the clean gap from {g0} to {g1} ({cut} smaller) while validation accuracy moved {dva}, inside the ±{band} noise band.",
               "Dropout at p = 0.2 did not deliver the predicted balance (gap {g0} → {g1}; validation {dva}, band ±{band})."),
    Hypothesis("H8", "Dropout's training accuracy is a measurement artifact",
               "At p = 0.4, Keras-reported training accuracy falls below validation accuracy in some epochs, yet clean training accuracy stays above it.",
               "At least one epoch with Keras training < validation, and clean training ≥ validation at the best epoch.", "Srivastava et al. (2014)", t_dropout_artifact,
               "Training accuracy below validation accuracy under dropout is a measurement effect: it happened in {n} epochs, but with dropout switched off training accuracy was {clean} against {val} on validation.",
               "The dropout measurement effect did not appear as predicted ({n} epochs; clean {clean} vs. validation {val})."),
    Hypothesis("H9", "Overfitting shows in calibration first",
               "Between the best epoch and the last epoch, ECE grows more for the unregularized He model than for the best regularized model.",
               "Δ ECE (He) > Δ ECE (regularized).", "Guo et al. (2017); Naeini et al. (2015)", t_calibration,
               "Continued training without regularization degraded calibration more ({d_he} ECE for He vs. {d_reg} for {reg}).",
               "Calibration did not degrade more for the unregularized model ({d_he} vs. {d_reg} for {reg})."),
    Hypothesis("H10", "L2 suppresses uninformative inputs",
               "Under L2 (λ = 0.001), first-layer weight magnitudes align more closely with per-pixel variability than without it.",
               "Pearson r(weight map, pixel SD) higher for L2 0.001 than for He.", "Krogh & Hertz (1991)", t_l2_inputs,
               "The penalty removed weight from pixels that carry no information: the alignment between weight magnitude and pixel variability rose from r = {r_he} to r = {r_l2}.",
               "L2 did not align the weights with pixel variability (r = {r_he} → {r_l2})."),
    Hypothesis("H11", "Errors follow visual similarity",
               "Class pairs whose average images are more alike are confused more often on the test set.",
               "Spearman ρ > 0.3 with p < 0.05 across the 45 class pairs.", "Xiao et al. (2017)", t_errors,
               "Errors are concentrated where garments genuinely look alike (Spearman ρ = {rho}, p = {p}).",
               "Errors were not explained by prototype similarity (ρ = {rho}, p = {p})."),
    Hypothesis("H12", "Initializer matters least in the search",
               "In the hyperparameter search, the initializer has the lowest importance of the three hyperparameters.",
               "Lowest fANOVA importance.", "Hutter et al. (2014)", t_tuning,
               "The search agreed with the controlled experiments: the initializer was the least important hyperparameter ({imp}).",
               "The search disagreed: the initializer was not the least important hyperparameter ({imp})."),
    Hypothesis("H13", "Validation-based selection transfers",
               "The selected model's test accuracy is within 1 pp of its best validation accuracy.",
               "|test − validation| < 1 pp.", "Cawley & Talbot (2010)", t_transfer,
               "The model chosen on validation data transferred to unseen data: {val} on validation, {test} on test ({d}; 95% interval {lo} to {hi}).",
               "The validation choice transferred less well than expected ({val} → {test}, {d})."),
]


def evaluate(A) -> list[Result]:
    out = []
    for h in HYPOTHESES:
        try:
            ok, evidence, values = h.test(A)
            out.append(Result(h, SUPPORTED if ok else NOT_SUPPORTED, evidence, values))
        except (LookupError, KeyError, TypeError, ZeroDivisionError) as err:
            out.append(Result(h, INCONCLUSIVE, f"{type(err).__name__}: {err}"))
    return out
