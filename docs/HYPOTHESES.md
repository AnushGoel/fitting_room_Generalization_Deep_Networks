# Pre-registered hypotheses

**Registered:** 7 October 2026, before the full experiment (all six configurations, three seeds, and the search) was run.
**Status:** frozen. Any later change is recorded in the deviations log at the bottom, with the reason.

## Why pre-register a small study

With six configurations, a dozen diagnostics, and three seeds, there are enough numbers to tell almost any story after the
fact. Writing the predictions and their decision rules down first separates the claims this project set out to test from
patterns noticed along the way (Nosek et al., 2018). It also makes failed predictions visible: a rule that fails is reported
as "not supported" rather than quietly dropped.

## Protocol the hypotheses refer to

Fashion-MNIST, a stratified 50,000 / 10,000 train–validation split with seed 42, the original 10,000 test images held out.
A fixed 784–128–128–10 ReLU network trained with Adam (default settings), sparse categorical cross-entropy, batch size 128,
and 40 epochs. Metrics are read at the epoch of best validation accuracy. "Clean" training accuracy is measured on a fixed
10,000-image training subset at the end of each epoch with dropout off. The sampling band is
1.96 · √(2p(1 − p)/n) with n = 10,000 (about ±0.87 pp at p = 0.89). Full details are in [METHODOLOGY.md](METHODOLOGY.md).

## Hypotheses and decision rules

| ID | Topic | Prediction | Decision rule | Basis |
|---|---|---|---|---|
| H1 | Signal propagation | At initialization, the ratio of second-hidden-layer activation scale under Glorot vs. He matches the analytical prediction. | Measured ratio within ±15% of the predicted ratio. | He et al. (2015); Glorot & Bengio (2010) |
| H2 | Initialization acts early | The effect of He vs. Glorot initialization on validation accuracy is larger in epochs 1–5 than at the best epoch. | |Δ mean validation accuracy, epochs 1–5| > |Δ best validation accuracy|. | Kingma & Ba (2015); He et al. (2015) |
| H3 | Initialization and final accuracy | He initialization does not change best validation accuracy beyond run-to-run noise in this two-layer network. | |Δ| below twice the larger seed SD (or below the sampling band with one seed). | Bouthillier et al. (2021) |
| H4 | Baseline overfits in confidence | Without regularization, validation cross-entropy rises after an early minimum while validation accuracy stays near its peak. | Validation CE ends >10% above its minimum and final validation accuracy is <1 pp below its best. | Guo et al. (2017) |
| H5 | Strong L2 underfits | L2 with λ = 0.01 lowers both clean training accuracy and best validation accuracy relative to He. | Clean training accuracy down by >2 pp and validation accuracy down by more than the sampling band. | Krogh & Hertz (1991) |
| H6 | Moderate L2 narrows the gap | L2 with λ = 0.001 reduces the clean train–validation gap by at least 30% relative to He. | Clean gap at the best epoch at most 70% of He's. | Krogh & Hertz (1991); Goodfellow et al. (2016) |
| H7 | Dropout balances fit and generalization | Dropout at p = 0.2 reduces the clean gap by at least 30% without lowering best validation accuracy beyond noise. | Clean gap at most 70% of He's and Δ validation accuracy above −(sampling band). | Srivastava et al. (2014) |
| H8 | Dropout's training accuracy is a measurement artifact | At p = 0.4, Keras-reported training accuracy falls below validation accuracy in some epochs, yet clean training accuracy stays above it. | At least one epoch with Keras training < validation, and clean training ≥ validation at the best epoch. | Srivastava et al. (2014) |
| H9 | Overfitting shows in calibration first | Between the best epoch and the last epoch, ECE grows more for the unregularized He model than for the best regularized model. | Δ ECE (He) > Δ ECE (regularized). | Guo et al. (2017); Naeini et al. (2015) |
| H10 | L2 suppresses uninformative inputs | Under L2 (λ = 0.001), first-layer weight magnitudes align more closely with per-pixel variability than without it. | Pearson r(weight map, pixel SD) higher for L2 0.001 than for He. | Krogh & Hertz (1991) |
| H11 | Errors follow visual similarity | Class pairs whose average images are more alike are confused more often on the test set. | Spearman ρ > 0.3 with p < 0.05 across the 45 class pairs. | Xiao et al. (2017) |
| H12 | Initializer matters least in the search | In the hyperparameter search, the initializer has the lowest importance of the three hyperparameters. | Lowest fANOVA importance. | Hutter et al. (2014) |
| H13 | Validation-based selection transfers | The selected model's test accuracy is within 1 pp of its best validation accuracy. | |test − validation| < 1 pp. | Cawley & Talbot (2010) |

## Analysis plan

1. Each rule is evaluated by `python -m fitting_room hypotheses` on the exported artifacts; the implementation is
   `src/fitting_room/hypotheses.py`, and this table is generated from the same code.
2. A hypothesis whose inputs are missing (for example, the seed study switched off) is reported as **inconclusive**, never as
   supported.
3. Thresholds were set from the analytic results in METHODOLOGY.md (Sections 6–8) before any full run, and are not tuned
   afterward. If a threshold turns out to be badly chosen, the original verdict stays in the report and the change is logged below.
4. Only H13 touches the test set, through the single evaluation of the model selected on validation data.

## Deviations log

| Date | Change | Reason |
|---|---|---|
| | *none so far* | |

## Reference

Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the
National Academy of Sciences, 115*(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114
