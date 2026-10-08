# Datasheet: Fashion-MNIST as used in this project

Following the questions proposed by Gebru et al. (2021). Facts about the dataset itself come from its authors (Xiao et al., 2017); choices about how it is used here are this project's own.

## Motivation

Fashion-MNIST was released by Zalando Research as a more challenging drop-in replacement for the MNIST handwritten digits, with the same image size, format, and train/test sizes, so that existing pipelines can switch without changes (Xiao et al., 2017). This project uses it because it is small enough to train many controlled runs on a laptop, yet hard enough that generalization is not trivial.

## Composition

- 70,000 grayscale images of 28 × 28 pixels, 8-bit intensities.
- 10 classes, 7,000 images each: T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot.
- Official split: 60,000 training and 10,000 test images, balanced across classes.
- Each instance is a single product image with one label. There are no people, faces, or personal information.

## Collection and preprocessing (by the dataset authors)

The images were derived from product thumbnails in Zalando's online catalogue. According to the authors, each thumbnail was trimmed, resized so that its longer edge is 28 pixels, sharpened, centered on a 28 × 28 canvas, inverted so that garments appear light on a dark background, and converted to 8-bit grayscale (Xiao et al., 2017).

## Preprocessing in this project

- Flattened to 784 values and divided by 255 (no per-pixel standardization; see METHODOLOGY.md, Section 3).
- The original training set is split 50,000 / 10,000 into training and validation with a stratified split (seed 42). The split indices are saved to `artifacts/split_indices.npz` with a fingerprint printed by the notebook.
- The original test set is used once, for the final evaluation.

## Known limitations

- Several upper-body classes have nearly identical average images, so a share of their errors reflects ambiguity in the data rather than in the model.
- All images are centered catalogue shots on plain backgrounds. Performance on real-world photographs is not representative.
- A sizeable share of the pixels, mostly along the border, barely vary across the dataset and carry little information (the notebook's Figure 4 counts them).

## Uses

Benchmarking and teaching. It is not appropriate as a proxy for real-world fashion recognition.

## Distribution and license

Fashion-MNIST is distributed by Zalando Research under the MIT License and is downloaded automatically through `tensorflow.keras.datasets.fashion_mnist`. This repository does not redistribute the images; the dashboard's exported subsets are derived from them for visualization only.

## References

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86–92. https://doi.org/10.1145/3458723

Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms* (arXiv:1708.07747). arXiv. https://doi.org/10.48550/arXiv.1708.07747
