# Model card: The Fitting Room selected classifier

Structured after Mitchell et al. (2019). The metrics block is written by `python -m fitting_room report`; everything else is maintained by hand.

## Model details

- **Architecture:** fully connected network, 784 → 128 (ReLU) → 128 (ReLU) → 10 (softmax), 118,282 parameters.
- **Training:** Adam with default settings, sparse categorical cross-entropy, batch size 128, up to 40 epochs; weights restored from the epoch with the best validation accuracy.
- **Configuration:** chosen on validation data from three candidates (He initialization without regularization, the better L2 model, the better dropout model). See [docs/METHODOLOGY.md](docs/METHODOLOGY.md), Section 7.
- **Framework:** TensorFlow / Keras. Exported weights also run in plain NumPy (`fitting_room.analysis.forward`).
- **Author:** Anush Goel. **License:** MIT.

## Intended use

A reference model for studying generalization: how initialization, weight penalties, dropout, and early stopping change what a fixed network learns. It is suitable for teaching, for experimentation, and as a baseline in comparisons on Fashion-MNIST.

**Out of scope.** Classifying real product photographs, any commercial or automated decision about people or products, and any setting where errors carry a cost. The model has only seen centered, 28 × 28, inverted grayscale catalogue thumbnails; the dashboard's upload feature exists to show how quickly performance falls on anything else, not to suggest it works there.

## Metrics

<!-- METRICS:START -->
- **Selected configuration:** He + dropout (p = 0.2) (784-128-128-10 ReLU network, softmax output; hidden-layer initializer: he_normal; L2 on hidden kernels: none; dropout after each hidden layer: 0.2; Adam (default settings), batch size 128; weights restored from epoch 40 of 40)
- **Validation accuracy (best epoch 40):** 90.25%
- **Test accuracy:** 88.86% (95% Wilson interval 88.23% to 89.46%)
- **Test cross-entropy:** 0.3423
- **Test expected calibration error:** 2.87%
- **Weakest classes by F1:** Shirt (0.708), Pullover (0.799), Coat (0.808)
- **Strongest classes by F1:** Sandal (0.973), Bag (0.975), Trouser (0.981)
<!-- METRICS:END -->

Accuracy is the headline metric because the classes are balanced. It is reported with a Wilson interval, alongside per-class precision, recall, and F1, cross-entropy, and expected calibration error, because a model can hold its accuracy while its probabilities become overconfident (Guo et al., 2017).

## Evaluation data

The original 10,000-image Fashion-MNIST test set (1,000 per class), used exactly once after the model was selected. Model selection used a stratified 10,000-image validation split of the training set.

## Training data

50,000 images from the Fashion-MNIST training set, pixel values scaled to [0, 1]. See [DATASHEET.md](DATASHEET.md).

## Factors and known failure modes

- **Visually similar classes.** Most errors fall among T-shirt/top, shirt, pullover, and coat, whose average images are nearly identical. Some of these errors reflect genuine ambiguity in the data.
- **Position sensitivity.** A dense layer has no built-in notion of translation; shifting a garment by a few pixels lowers accuracy sharply (see the robustness sweep in the dashboard).
- **Distribution shift.** Real photographs differ in background, lighting, pose, and framing, and are classified far less reliably.

## Ethical considerations

The data are product images with no people's faces or personal information. The main risk is over-trust: a confident prediction on an out-of-distribution image is still likely to be wrong. Calibration is reported for that reason, and post-hoc temperature scaling is recommended before any probability is used for a decision.

## Caveats

Results come from one data split and three training seeds. Differences of a few tenths of a percentage point between configurations are within run-to-run noise (Bouthillier et al., 2021).

## References

Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Mohammadi Sepahvand, N., Raff, E., Madan, K., Voleti, V., Ebrahimi Kahou, S., Michalski, V., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems, 3*.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321–1330). PMLR.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). https://doi.org/10.1145/3287560.3287596
