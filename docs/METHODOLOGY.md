# Methodology: what was done, and why

This document explains every design decision in the study, the theory behind each technique, and how the conclusions are proven. The results themselves live in [`RESULTS.md`](RESULTS.md), which is generated from the experiment outputs so that the numbers there cannot drift from the data. The predictions those results were tested against were written down in advance in [`HYPOTHESES.md`](HYPOTHESES.md).

## 1. The question

Deep networks can fit their training data almost perfectly, so the interesting question is rarely whether a network *can* fit, but how much of what it fits will hold on data it has never seen. A handful of standard techniques are said to help: careful weight initialization, L2 regularization, dropout, and early stopping. In practice they are usually switched on together, which makes it hard to say what any one of them actually does.

This project takes the opposite approach. One small network, one dataset, one optimizer, one training budget, and one data split are held fixed, and a single technique is changed at a time. Any difference in behavior can then be attributed to that technique. The questions are deliberately narrow:

1. Does He initialization change how a ReLU network trains, how well it generalizes, or both?
2. How does the strength of an L2 penalty trade training fit against generalization, and where does it tip into underfitting?
3. Does dropout narrow the train–validation gap without costing accuracy, and why can its training accuracy fall below its validation accuracy?
4. Which configuration gives the best balance, and how confident can we be in that choice?

## 2. Design principles

**One factor at a time.** Each configuration differs from its parent in exactly one respect (Figure 1 of the notebook). The baseline uses the Keras defaults; the second model changes only the initializer; every regularized model starts from the second model and adds one regularizer at one strength. This is a classic controlled design: it gives up the ability to study interactions in exchange for clean attribution, and the interaction question is handed to a separate hyperparameter search (Section 9).

**Fixed protocol.** Architecture (784–128–128–10, ReLU, softmax), optimizer (Adam with default settings), loss (sparse categorical cross-entropy), batch size (128), epoch budget (40), and data split are identical everywhere. Adam and its learning rate are intentionally not tuned. Changing the optimizer changes the effective strength of regularization (Section 6.2), so leaving it free would confound every comparison.

**Selection on validation, one look at the test set.** All decisions are made on a validation set carved out of the training images. The test set is evaluated exactly once, on a model whose configuration and weights were frozen beforehand. Choosing a model is itself a form of fitting, and every comparison made on a dataset biases the score of the winner on that dataset upward; the more candidates, the larger the bias (Cawley & Talbot, 2010). Keeping the test set out of every decision is what makes its single score an honest estimate.

**Predictions before results.** Thirteen hypotheses, each with a numeric decision rule, were written before the full experiment ran. Fixing the rule in advance separates confirmatory claims from the patterns one inevitably finds after looking at the data (Nosek et al., 2018). The verdicts are computed by code (`src/fitting_room/hypotheses.py`), not judged by eye.

**Variance is part of the result.** A single training run is one draw from a distribution over initial weights, mini-batch orders, and dropout masks. Bouthillier et al. (2021) show that this variation alone can reverse published conclusions. Every configuration is therefore repeated with additional seeds on the same data split, and paired significance tests are used where models share the same validation images.

## 3. Data

Fashion-MNIST contains 70,000 grayscale 28 × 28 images of clothing in 10 balanced classes, 60,000 for training and 10,000 for testing (Xiao et al., 2017). It was built as a harder drop-in replacement for handwritten digits. Several classes (T-shirt/top, shirt, pullover, coat) are visually close, which matters for interpretation: some fraction of errors reflects genuine ambiguity in the data rather than a failure of training. The [datasheet](../DATASHEET.md) describes the dataset in more detail.

**Split.** The 60,000 original training images are divided into 50,000 for training and 10,000 for validation with a stratified, seeded split, so each class keeps its 10% share in every partition. The split is done on indices and saved with a fingerprint, so it can be audited and reproduced.

**Preprocessing.** Each image is flattened to 784 values and divided by 255. Three reasons make normalization necessary rather than cosmetic. The initialization schemes in Section 6.1 assume inputs of modest, roughly unit scale (Glorot & Bengio, 2010; He et al., 2015). A small, common input scale gives a better-conditioned optimization problem (LeCun et al., 1998). And an L2 penalty is scale dependent, so λ only has a stable meaning once the inputs have a fixed scale. Per-pixel standardization was rejected because a large share of the border pixels is almost constant; dividing by their near-zero standard deviation would turn noise into large inputs.

## 4. The model

The network has two hidden layers of 128 ReLU units and a 10-way softmax output. It has

> 784·128 + 128 + 128·128 + 128 + 128·10 + 10 = 100,480 + 16,512 + 1,290 = **118,282** parameters,

more than two per training image. That is ample capacity to memorize; Zhang et al. (2017) showed that networks of this kind can fit even random labels. Overfitting is therefore possible, but whether it actually occurs in 40 epochs is an empirical question, not an assumption.

## 5. What is measured, beyond what Keras reports

Keras reports training and validation loss and accuracy. Each of those has a blind spot for this study, so a custom callback records additional quantities after every epoch, plus a reading of the untrained network as epoch 0:

| Quantity | Why it is needed |
|---|---|
| Training accuracy with dropout off, on a fixed 10,000-image subset, at the end of the epoch | Keras' training accuracy is measured with dropout active and averaged over the epoch while weights change. Comparing it with validation accuracy compares two different networks. |
| Validation cross-entropy without the L2 term | Keras adds λΣW² to every reported loss, so the loss of an L2 model is not comparable with the loss of any other model. |
| Squared weight norm of each layer, and the size of the penalty | Shows directly whether L2 is doing what it is supposed to do. |
| Share of hidden units that never activate on 2,000 validation images | Detects "dead" ReLU units, the typical failure mode of a poor initialization. |
| RMS of hidden activations | Tests the variance-propagation theory in Section 6.1 against measurement. |
| Weights at initialization, at the best validation epoch, and at the end | Allow early stopping to be replayed offline and calibration to be compared at two points in training. |

The reporting rule is the same for every model: metrics are read at the epoch with the best validation accuracy.

## 6. The techniques, and what theory predicts

### 6.1 Weight initialization

For a layer with zero-mean, symmetric weights of variance Var(W) and zero biases, the second moment of each pre-activation is

> E[z²] = fan_in · Var(W) · E[h²],

and a ReLU passes on half of it: E[ReLU(z)²] = E[z²] / 2 (He et al., 2015). He initialization chooses Var(W) = 2 / fan_in, so the factor fan_in · Var(W) / 2 equals 1 and the signal keeps its scale through every layer. The Keras default, Glorot uniform, uses Var(W) = 2 / (fan_in + fan_out), derived for symmetric activations (Glorot & Bengio, 2010). For this network the per-layer gains under Glorot are

> layer 1: 784 · (2/912) / 2 = 0.860, layer 2: 128 · (2/256) / 2 = 0.500,

so the second moment of the second hidden layer ends at 0.860 × 0.500 = **0.43** of the He value, and the activation scale at √0.43 = **0.66**. That is a mild shrinkage. With equal widths, each additional hidden-to-hidden layer halves the second moment again, so the same mismatch would leave about 10⁻⁶ of the second moment after 20 layers. The dashboard's initialization lab plots this curve for any width and depth.

**Prediction.** In a two-layer network the effect should be visible early in training and small at the end. Adam reinforces this: it divides each gradient by a running estimate of its own magnitude, so a layer whose gradients are smaller because of a narrower initialization still receives steps of similar size (Kingma & Ba, 2015). The prediction is tested three ways: the measured activation ratio against 0.66 (H1), the early versus final effect (H2), and the final effect against seed noise (H3).

### 6.2 L2 regularization

L2 adds λΣW² over the hidden kernels to the training loss. Its gradient, 2λW, pulls every weight toward zero in proportion to its size, so the optimizer settles for smaller weights and a smoother function (Krogh & Hertz, 1991). For a linear model, Krogh and Hertz proved that the penalty suppresses components of the weight vector the data do not determine. In this dataset, the near-constant border pixels are exactly such components, which leads to a testable prediction about where the weights shrink (H10).

**How strong is strong?** At He initialization the expected squared norm of the two hidden kernels is

> E[ΣW²] = 100,352 · (2/784) + 16,384 · (2/128) = 256 + 256 = **512**.

The untrained network's cross-entropy is about ln 10 ≈ 2.30. With λ = 0.01 the penalty therefore starts at about **5.1**, more than twice the data loss, and the optimizer's first priority becomes shrinking weights rather than fitting garments. With λ = 0.001 it starts at about **0.51**, the same order as the data loss. This simple ratio predicts underfitting at λ = 0.01 (H5) and a useful trade at λ = 0.001 (H6) before any training is done.

**A caveat about Adam.** An L2 term passed through Adam is not the same as weight decay: Adam rescales the penalty's gradient by each weight's gradient history, so frequently updated weights are decayed less than λ suggests (Loshchilov & Hutter, 2019). The λ values here are specific to Adam and would not transfer directly to SGD or AdamW.

### 6.3 Dropout

During training each hidden unit is zeroed with probability p and the survivors are scaled by 1/(1 − p), so nothing needs rescaling at inference, when dropout is off. Srivastava et al. (2014) interpret this as training an exponentially large family of thinned networks that share weights; with 256 hidden units there are 2²⁵⁶ possible masks, and the full network at test time approximates their average. Because no unit can rely on a particular partner being present, units must learn features that are useful on their own.

**Why training accuracy can sit below validation accuracy.** Three effects make the Keras training number pessimistic. Each training prediction comes from a thinned network missing 20% or 40% of its hidden units, while validation uses the complete network. Keras averages training accuracy over every mini-batch in the epoch, including early batches scored by weights that have not finished the epoch's learning, while validation is scored once at the end. And nothing makes the validation number optimistic: it is the network that is actually deployed. Measuring training accuracy the same way as validation accuracy, with dropout off and at the end of the epoch, should remove the paradox (H8).

**Prediction.** At p = 0.2 dropout should narrow the gap substantially without lowering validation accuracy beyond noise (H7). At p = 0.4, with only about 77 of 128 units active in each step, it may start to underfit.

### 6.4 Early stopping

Every run trains for the full 40 epochs and records its whole history, so `EarlyStopping` with `restore_best_weights=True` can be replayed offline for any patience and either monitor without retraining. Early stopping is a regularizer in its own right; for linear models trained by gradient descent it behaves much like an L2 penalty whose strength shrinks as training continues (Goodfellow et al., 2016). The practical choice it raises is the monitor: validation loss stops near the point of best calibration, validation accuracy near the point of best classification, and the two need not coincide (Prechelt, 1998).

## 7. Selection and the single test evaluation

The candidates are the He model without regularization, the better of the two L2 models, and the better of the two dropout models, each judged by best validation accuracy. The winner is the candidate with the highest best-epoch validation accuracy; an exact tie goes to the smaller gap between clean training and validation accuracy. The winner is restored to its best-epoch weights, which is what early stopping with weight restoration would return, and is evaluated once on the test set.

Test accuracy is reported with two intervals. The Wilson score interval has good coverage for proportions near the edges of [0, 1] (Wilson, 1927); at p ≈ 0.89 and n = 10,000 its half-width is about

> 1.96 · √(0.89 · 0.11 / 10,000) ≈ **0.61** percentage points.

A percentile bootstrap over the 10,000 per-image outcomes is reported alongside it as a distribution-free check (Efron & Tibshirani, 1993). A small drop from validation to test is expected rather than alarming: best-epoch validation accuracy is the maximum of 40 noisy measurements and is therefore optimistic by construction. The prediction that the drop stays under one point is H13.

## 8. Statistical inference

**How large a difference is meaningful?** The standard error of the difference between two independent accuracies near p on n images is √(2p(1 − p)/n). At p = 0.89 and n = 10,000 that gives a 95% band of

> 1.96 · √(2 · 0.89 · 0.11 / 10,000) ≈ **±0.87** percentage points.

Differences smaller than this between single runs are within sampling noise. The band is conservative here, because the models are scored on the same images.

**Paired tests.** Because every model is scored on the same 10,000 validation images, McNemar's test is the right comparison. It uses only the images on which two models disagree: *b* counts images the first model gets right and the second gets wrong, *c* the reverse, and under the null hypothesis of equal accuracy *b* and *c* follow a binomial distribution with p = ½ (Dietterich, 1998; McNemar, 1947). The exact binomial version is used, and p-values are adjusted for multiple comparisons with Holm's step-down procedure (Holm, 1979). One caveat matters: the selected model was chosen *because* it scored highest on this validation set, so these p-values are descriptive and somewhat favor the winner.

**Seeds.** All six configurations are repeated with two additional seeds on the same split, so only training randomness varies. A difference is treated as real only if it exceeds about twice the seed-to-seed standard deviation.

## 9. Hyperparameter search

The controlled experiments cannot show interactions, for example whether a little L2 combined with a little dropout beats either alone. A Bayesian search addresses that, restricted to the three techniques under study so that it stays inside the fixed protocol:

| Hyperparameter | Range | Reason |
|---|---|---|
| Initializer | Glorot uniform, He normal, He uniform | The default and the two ReLU-matched variants |
| L2 coefficient | 10⁻⁶ to 10⁻², log-uniform | Strength acts multiplicatively; log spacing covers orders of magnitude evenly (Bergstra & Bengio, 2012) |
| Dropout rate | 0 to 0.6, steps of 0.05 | Srivastava et al. (2014) found 0.5 near-optimal for wide layers; narrower layers should prefer less |

The search uses Optuna's tree-structured Parzen estimator, which models which regions of the space produce good results and samples there (Akiba et al., 2019; Bergstra et al., 2011), with a median pruner that stops trials falling behind earlier ones at the same epoch (Golovin et al., 2017). Hyperparameter importance is estimated with fANOVA (Hutter et al., 2014), which leads to H12. The tuned configuration is compared with the selected model on validation data and across seeds. It is deliberately **not** evaluated on the test set: testing a second, more heavily searched model and comparing the two scores would turn the test set into a selection tool.

## 10. Calibration

A model is calibrated when predictions made with confidence c are correct about c of the time. Expected calibration error (ECE) bins predictions by confidence, takes the gap between accuracy and mean confidence in each bin, and averages the gaps weighted by bin size (Naeini et al., 2015). Modern networks tend to become overconfident as they overfit, even while accuracy holds (Guo et al., 2017). That gives two predictions: the baseline's validation loss should rise while its accuracy stays flat (H4), and continued training should degrade calibration more without regularization than with it (H9).

## 11. Engineering for reproducibility

Reproducibility in machine learning fails less often because of mathematics than because of missing code, missing settings, or untracked runs (Pineau et al., 2021). The repository is organized so that each of those is covered:

- **One source of truth for the analysis.** Metric, test, and simulation code lives in a small NumPy-only package (`src/fitting_room/`) that the dashboard and the report builder both import. Its behavior is pinned down by unit tests, which include the analytic results in this document (the 0.43 ratio, the Holm adjustments, the exact binomial tail).
- **A shared run cache.** The notebook and the command-line trainer write each run to `artifacts/runs/<tag>/` in the same format, so a run trained on a server is reused by the notebook without retraining.
- **A run registry.** Every cached run is listed with a fingerprint of its configuration, its seed, and its provenance (source, git commit, library versions, timestamp), so any number in the report can be traced to the run that produced it.
- **Generated results.** `RESULTS.md` and the results block in the README are written by code from the artifacts.
- **Continuous integration.** Every push runs the linter and the test suite on Python 3.11 and 3.12 and checks that every notebook cell parses.
- **Storage discipline.** Heavy outputs (run cache, full-precision weights, the search database) are regenerable and excluded from version control. A slim export keeps what the dashboard needs in a few megabytes, and figures are downscaled and palette-quantized before they are committed. Keeping large binary outputs out of the code history is one of the cheapest ways to avoid the hidden maintenance costs of ML systems (Sculley et al., 2015).

## 12. Threats to validity

The usual four families of threats (Shadish et al., 2002) apply as follows.

**Internal validity** asks whether the technique, and not something else, caused the change. One-factor-at-a-time changes and a fixed protocol address this directly. The remaining confound is training randomness, which the seed study measures.

**Statistical conclusion validity** asks whether differences are real. Validation accuracy is noisy at the level of a few tenths of a point, best-epoch selection is optimistic, and the paired tests are run on the same data used for selection. These are handled with intervals, seeds, Holm adjustment, and explicit caveats rather than ignored.

**Construct validity** asks whether the measurements mean what they are taken to mean. The gap is measured with dropout off, losses are compared without the L2 term, and calibration is measured separately from accuracy, precisely because the default quantities would mislead.

**External validity** asks how far the results travel. They are specific to a shallow dense network, Adam at its default learning rate, and one dataset. The initialization result in particular is expected to change with depth and with SGD, and the dashboard's signal-propagation calculator shows by how much. Only one data split was used, so variance due to data sampling is not measured.

## 13. How the conclusions are proven

Two kinds of proof support the conclusions.

**Analytic results** are derived above and do not depend on the run: the 0.43 second-moment ratio of Glorot to He in this network, the starting penalty of about 5.1 versus a data loss of 2.3 at λ = 0.01, the three reasons dropout's training accuracy is pessimistic, and the ±0.6 and ±0.87 point noise bands. The numerical ones are checked by unit tests.

**Empirical results** are the verdicts of the thirteen pre-registered decision rules, evaluated automatically on the exported artifacts. Each verdict in `RESULTS.md` is printed together with the numbers it was computed from, so a reader can check the reasoning without rerunning anything, and anyone who reruns the pipeline gets a fresh report from their own run. Where a prediction fails, the report says so, because a failed prediction is evidence too.

## References

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631). https://doi.org/10.1145/3292500.3330701

Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. In *Advances in Neural Information Processing Systems* (Vol. 24, pp. 2546–2554). Curran Associates.

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305.

Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Mohammadi Sepahvand, N., Raff, E., Madan, K., Voleti, V., Ebrahimi Kahou, S., Michalski, V., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems, 3*. https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107.

Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923. https://doi.org/10.1162/089976698300017197

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.

Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics* (Vol. 9, pp. 249–256). PMLR. https://proceedings.mlr.press/v9/glorot10a.html

Golovin, D., Solnik, B., Moitra, S., Kochanski, G., Karro, J., & Sculley, D. (2017). Google Vizier: A service for black-box optimization. In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1487–1495). https://doi.org/10.1145/3097983.3098043

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321–1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 1026–1034). https://doi.org/10.1109/ICCV.2015.123

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.

Hutter, F., Hoos, H., & Leyton-Brown, K. (2014). An efficient approach for assessing hyperparameter importance. In *Proceedings of the 31st International Conference on Machine Learning* (Vol. 32, pp. 754–762). PMLR. https://proceedings.mlr.press/v32/hutter14.html

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations*. https://arxiv.org/abs/1412.6980

Krogh, A., & Hertz, J. A. (1991). A simple weight decay can improve generalization. In J. Moody, S. Hanson, & R. P. Lippmann (Eds.), *Advances in Neural Information Processing Systems* (Vol. 4, pp. 950–957). Morgan Kaufmann.

LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). Efficient BackProp. In G. B. Orr & K.-R. Müller (Eds.), *Neural networks: Tricks of the trade* (pp. 9–50). Springer. https://doi.org/10.1007/3-540-49430-8_2

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *Proceedings of the 7th International Conference on Learning Representations*. https://arxiv.org/abs/1711.05101

McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157. https://doi.org/10.1007/BF02295996

Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. In *Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence* (pp. 2901–2907). https://doi.org/10.1609/aaai.v29i1.9602

Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences, 115*(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114

Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). Improving reproducibility in machine learning research (a report from the NeurIPS 2019 Reproducibility Program). *Journal of Machine Learning Research, 22*(164), 1–20.

Prechelt, L. (1998). Early stopping—But when? In G. B. Orr & K.-R. Müller (Eds.), *Neural networks: Tricks of the trade* (pp. 55–69). Springer. https://doi.org/10.1007/3-540-49430-8_3

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. In *Advances in Neural Information Processing Systems* (Vol. 28, pp. 2503–2511). Curran Associates.

Shadish, W. R., Cook, T. D., & Campbell, D. T. (2002). *Experimental and quasi-experimental designs for generalized causal inference*. Houghton Mifflin.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953

Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms* (arXiv:1708.07747). arXiv. https://doi.org/10.48550/arXiv.1708.07747

Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. In *Proceedings of the 5th International Conference on Learning Representations*. https://arxiv.org/abs/1611.03530
