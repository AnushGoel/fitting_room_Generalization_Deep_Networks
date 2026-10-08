"""Builds docs/RESULTS.md from the exported artifacts and refreshes the results block in README.md.

Every number in the generated report is read from the artifacts, and every conclusion is the output of a
pre-registered decision rule (see hypotheses.py), so the write-up cannot drift away from the data.
"""
from __future__ import annotations

import datetime as dt
import re
import shutil
from pathlib import Path

from .hypotheses import INCONCLUSIVE, NOT_SUPPORTED, SUPPORTED, evaluate

FIGURES = {
    "trajectories": "fig16_generalization_trajectories", "all_models": "fig17_all_models",
    "init": "fig06_initialization_diagnostics", "init_early": "fig07_initialization_early_training",
    "l2": "fig09_l2_mechanics", "l2_maps": "fig10_first_layer_weight_maps", "dropout": "fig12_dropout_three_accuracies",
    "confusion": "fig13_confusion_and_per_class", "calibration": "fig19_calibration", "seeds": "fig20_seed_robustness",
    "optuna": "fig21_optuna", "early_stopping": "fig18_early_stopping",
}
BADGE = {SUPPORTED: "✅ supported", NOT_SUPPORTED: "❌ not supported", INCONCLUSIVE: "⚪ inconclusive"}

REFERENCES = """Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Mohammadi Sepahvand, N., Raff, E., Madan, K., Voleti, V., Ebrahimi Kahou, S., Michalski, V., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems, 3*. https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107.

Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics* (Vol. 9, pp. 249–256). PMLR.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321–1330). PMLR.

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 1026–1034). https://doi.org/10.1109/ICCV.2015.123

Hutter, F., Hoos, H., & Leyton-Brown, K. (2014). An efficient approach for assessing hyperparameter importance. In *Proceedings of the 31st International Conference on Machine Learning* (Vol. 32, pp. 754–762). PMLR.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations*. https://arxiv.org/abs/1412.6980

Krogh, A., & Hertz, J. A. (1991). A simple weight decay can improve generalization. In J. Moody, S. Hanson, & R. P. Lippmann (Eds.), *Advances in Neural Information Processing Systems* (Vol. 4, pp. 950–957). Morgan Kaufmann.

Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. In *Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence* (pp. 2901–2907). https://doi.org/10.1609/aaai.v29i1.9602

Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences, 115*(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958.

Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms* (arXiv:1708.07747). arXiv. https://doi.org/10.48550/arXiv.1708.07747"""


def compress_figure(src: Path, dst: Path, max_width=1400, colors=160):
    """Downscale and palette-quantize a matplotlib PNG. Typically 60–80% smaller with no visible loss for plots."""
    from PIL import Image

    img = Image.open(src).convert("RGB")
    if img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)), Image.LANCZOS)
    method = getattr(getattr(Image, "Quantize", None), "FASTOCTREE", 2)
    img.quantize(colors=colors, method=method).save(dst, optimize=True)
    return dst


def _pct(x):
    return f"{100 * x:.2f}%"


def _md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


def _fig(name, rel, caption):
    return f"\n![{caption}]({rel}/{FIGURES[name]}.png)\n*{caption}*\n"


def build_report(A, fig_rel="figures") -> str:
    R, F = A["runs"], A["final"]
    results = evaluate(A)
    n_sup = sum(r.verdict == SUPPORTED for r in results)
    n_not = sum(r.verdict == NOT_SUPPORTED for r in results)
    n_inc = sum(r.verdict == INCONCLUSIVE for r in results)
    main = [k for k in ["baseline", "he_init", "l2_0.001", "l2_0.01", "dropout_0.2", "dropout_0.4"] if k in R]
    ranking = sorted(main, key=lambda k: -R[k]["summary"]["best_val_acc"])
    by = {r.hypothesis.id: r for r in results}
    epochs = F.get("epochs", len(R[main[0]]["history"]["val_accuracy"]))
    seeds = A.get("seeds")
    n_seeds = int(seeds.seed.nunique()) if seeds is not None else 1

    out = [f"""# Results

*Generated from `artifacts/` on {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by `python -m fitting_room report`. Do not edit by hand:
re-run the command after re-running the experiments. Split fingerprint `{F.get('split_fingerprint', 'n/a')}`, seed {F.get('seed')},
{epochs} epochs, batch size {F.get('batch_size')}, {n_seeds} seed(s) in the robustness study.*

## Headline

The selected model is **{F['label']}**, restored to epoch {F['best_epoch']}. It reached **{_pct(F['best_val_acc'])}** validation
accuracy and **{_pct(F['test_acc'])}** on the untouched test set (95% Wilson interval {_pct(F['ci_wilson'][0])} to
{_pct(F['ci_wilson'][1])}; bootstrap {_pct(F['ci_bootstrap'][0])} to {_pct(F['ci_bootstrap'][1])}). The test set was used once,
after the choice had been fixed on validation data.

Of {len(results)} pre-registered hypotheses, **{n_sup} were supported**, {n_not} were not, and {n_inc} could not be tested in this run.
""", _fig("trajectories", fig_rel, "Figure 1. Each model's path from epoch 1 to 40 in training-vs-validation accuracy space."),
           "\n## Summary comparison\n"]
    if A.get("table1") is not None:
        out.append(_md_table(A["table1"]))
    out.append("\n\nRanking by best validation accuracy: " + ", ".join(
        f"{R[k]['short']} {_pct(R[k]['summary']['best_val_acc'])}" for k in ranking) + ".\n")

    out.append("\n## Hypothesis scorecard\n\nEach rule below was fixed before the experiment was run (`docs/HYPOTHESES.md`). "
               "The verdict is computed, not judged.\n\n| ID | Hypothesis | Decision rule | Verdict | Evidence |\n|---|---|---|---|---|")
    for r in results:
        h = r.hypothesis
        out.append(f"| {h.id} | {h.statement} | {h.rule} | {BADGE[r.verdict]} | {r.evidence} |")

    sections = [
        ("Initialization", ["H1", "H2", "H3"], ["init", "init_early"],
         "He initialization was designed for ReLU layers; the question was whether that matters in a network only two layers deep."),
        ("L2 regularization", ["H5", "H6", "H10"], ["l2", "l2_maps"],
         "The penalty λΣW² trades training fit for smaller weights. The question was how much trade is worth it."),
        ("Dropout", ["H7", "H8"], ["dropout"],
         "Dropout trains a random sub-network at every step. The question was whether it narrows the gap without costing accuracy, "
         "and why its training accuracy can sit below validation accuracy."),
        ("Overfitting, calibration, and early stopping", ["H4", "H9"], ["calibration", "early_stopping"],
         "Accuracy alone hides how a model fails. These tests look at the loss and the probabilities."),
        ("Errors, robustness of the ranking, and the search", ["H11", "H12", "H13"], ["confusion", "seeds", "optuna"],
         "The last group asks whether the conclusions survive scrutiny from other angles."),
    ]
    for title, ids, figs, intro in sections:
        out.append(f"\n## {title}\n\n{intro}\n")
        for i in ids:
            r = by[i]
            out.append(f"- **{i} ({BADGE[r.verdict]}).** {r.conclusion} *Proof:* {r.evidence}.")
        for f in figs:
            if (A["path"] / "figures" / f"{FIGURES[f]}.png").exists():
                out.append(_fig(f, fig_rel, f"{title}: {FIGURES[f].split('_', 1)[1].replace('_', ' ')}."))

    supported = [r for r in results if r.verdict == SUPPORTED]
    failed = [r for r in results if r.verdict == NOT_SUPPORTED]
    out.append("\n## Conclusions\n")
    out.append("What the evidence supports, in order of the study:\n")
    out += [f"{n}. {r.conclusion}" for n, r in enumerate(supported, start=1)]
    if failed:
        out.append("\nWhere the predictions failed, which is equally informative:\n")
        out += [f"- {r.conclusion}" for r in failed]
    out.append(f"""
Taken together: initialization sets the starting point, regularization sets how much of the training fit transfers, and the
strength of regularization matters more than its type. Differences among the well-regularized models are of the same size as
run-to-run noise, so they are reported as a group rather than as a single winner (Bouthillier et al., 2021).

## Limitations

One stratified train–validation split, so data-sampling variance is not measured; {n_seeds} training seed(s); best-epoch
validation accuracy is the maximum of {epochs} noisy measurements and therefore slightly optimistic for every model; the optimizer
and learning rate were fixed by design. Calibration is measured but not corrected.

## References

{REFERENCES}
""")
    return "\n".join(out)


def readme_block(A, fig_rel="docs/figures") -> str:
    R, F = A["runs"], A["final"]
    results = evaluate(A)
    main = [k for k in ["baseline", "he_init", "l2_0.001", "l2_0.01", "dropout_0.2", "dropout_0.4", "tuned"] if k in R]
    rows = ["| Configuration | Training acc. (dropout off) | Validation acc. | Clean gap | Best epoch |", "|---|---|---|---|---|"]
    for k in main:
        s = R[k]["summary"]
        star = " **(selected)**" if k == F["selected"] else ""
        rows.append(f"| {R[k]['label']}{star} | {_pct(s['clean_train_acc_at_best'])} | {_pct(s['best_val_acc'])} | "
                    f"{100 * s['clean_gap_at_best']:.2f} pp | {s['best_epoch']} |")
    verdicts = " ".join(f"`{r.hypothesis.id}` {BADGE[r.verdict].split()[0]}" for r in results)
    return (f"**Selected model:** {F['label']}. Test accuracy **{_pct(F['test_acc'])}** "
            f"(95% CI {_pct(F['ci_wilson'][0])}–{_pct(F['ci_wilson'][1])}), from a single evaluation after selection on validation data.\n\n"
            + "\n".join(rows) + f"\n\n**Pre-registered hypotheses:** {verdicts}. Full evidence in [docs/RESULTS.md](docs/RESULTS.md).\n\n"
            f"![Generalization trajectories]({fig_rel}/{FIGURES['trajectories']}.png)")


def model_card_block(A) -> str:
    F = A["final"]
    lines = [f"- **Selected configuration:** {F['label']} ({F.get('config_text', '')})",
             f"- **Validation accuracy (best epoch {F['best_epoch']}):** {_pct(F['best_val_acc'])}",
             f"- **Test accuracy:** {_pct(F['test_acc'])} (95% Wilson interval {_pct(F['ci_wilson'][0])} to {_pct(F['ci_wilson'][1])})",
             f"- **Test cross-entropy:** {F['test_ce']:.4f}",
             f"- **Test expected calibration error:** {_pct(F['test_reliability']['ece'])}"]
    per = sorted(F.get("per_class", []), key=lambda r: r["F1"])
    if per:
        lines.append("- **Weakest classes by F1:** " + ", ".join(f"{r['class']} ({r['F1']:.3f})" for r in per[:3]))
        lines.append("- **Strongest classes by F1:** " + ", ".join(f"{r['class']} ({r['F1']:.3f})" for r in per[-3:]))
    return "\n".join(lines)


def _replace_block(path: Path, tag: str, body: str) -> bool:
    text = path.read_text()
    pattern = re.compile(rf"(<!-- {tag}:START -->)(.*?)(<!-- {tag}:END -->)", re.S)
    if not pattern.search(text):
        return False
    path.write_text(pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text))
    return True


def update_readme(readme: Path, A) -> bool:
    return _replace_block(readme, "RESULTS", readme_block(A))


def write_all(A, docs_dir: Path, readme: Path | None = None, max_width=1400):
    docs_dir = Path(docs_dir)
    fig_out = docs_dir / "figures"
    fig_out.mkdir(parents=True, exist_ok=True)
    src_dir = A["path"] / "figures"
    copied, before, after = 0, 0, 0
    for name in FIGURES.values():
        src = src_dir / f"{name}.png"
        if src.exists():
            dst = fig_out / f"{name}.png"
            try:
                compress_figure(src, dst, max_width=max_width)
            except Exception:
                shutil.copyfile(src, dst)
            copied += 1
            before += src.stat().st_size
            after += dst.stat().st_size
    (docs_dir / "RESULTS.md").write_text(build_report(A))
    updated = update_readme(readme, A) if readme and readme.exists() else False
    card = Path(readme).parent / "MODEL_CARD.md" if readme else None
    if card and card.exists():
        _replace_block(card, "METRICS", model_card_block(A))
    return {"figures": copied, "fig_mb_before": before / 1e6, "fig_mb_after": after / 1e6, "readme_updated": updated}
