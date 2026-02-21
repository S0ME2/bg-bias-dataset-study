# Dataset Analysis — Oxford-IIIT Pet Dataset

## Dataset Overview

Name: Oxford-IIIT Pet Dataset
Source: [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)

Total images: 7390
Total segmentation masks: 7390

Training images: 3680
Test images: 3669

Segmentation masks available: Yes

---

## Task Definition

This research focuses on binary classification:

Classes:

* Cat
* Dog

The dataset originally contains 37 breeds:

* 12 cat breeds
* 25 dog breeds

For this research, breeds will be mapped into two superclass labels:

* Cat → label 0
* Dog → label 1

---

## Annotation Types Available

The dataset provides:

1. Image-level class labels
2. Pixel-level segmentation masks (trimaps)

Trimap pixel values:

* 1 = foreground (pet)
* 2 = background
* 3 = border

These masks enable controlled foreground-background separation.

---

## Image Characteristics

Image resolution:

* Variable resolution
* Approximate range: 200×200 to 500×500 pixels

Image format:

* JPEG

Background characteristics:

* Natural environments
* Indoor scenes
* Outdoor scenes
* Various textures, lighting, and context

This variability makes the dataset suitable for studying background bias.

---

## Dataset Integrity Verification

Verified properties:

* All images have corresponding segmentation masks
* Official train/test split available
* No missing or corrupted files

---

## Known Limitations

* Limited dataset size compared to modern vision datasets
* Binary classification simplifies real-world complexity
* Background manipulation in later phases introduces artificial patterns

These limitations are acceptable for controlled experimental study.

---

## Relevance to Research Question

This dataset is suitable for studying spurious background correlations because:

* Foreground can be cleanly separated
* Background can be independently manipulated
* Cat and dog classes provide a clear binary classification setting

This enables controlled counterfactual experiments.

---

## Binary Mask Construction

### Objective

The original Oxford-IIIT Pet Dataset provides segmentation trimaps with three pixel values:

* 1 = foreground (pet)
* 2 = background
* 3 = border

For the purposes of controlled background manipulation, these trimaps were converted into binary masks representing foreground and background regions.

---

### Method

Each trimap was converted into a binary mask using the following rule:

* Foreground pixels: trimap value ∈ {1, 3} → mask value = 1
* Background pixels: trimap value = 2 → mask value = 0

Border pixels were included as foreground to ensure full preservation of the animal and avoid losing fine structural details.

Binary masks were saved as PNG images with pixel values:

* 255 = foreground
* 0 = background

Total binary masks created: 7390

Each mask corresponds exactly to one image, preserving dataset alignment.

---

### Reason for This Step

This step enables explicit separation of foreground and background regions, which is required to construct controlled experimental datasets with manipulated backgrounds.

Specifically, binary masks allow:

* Removal of original backgrounds
* Replacement with controlled artificial backgrounds
* Creation of background-biased and counterfactual datasets

This is essential for isolating background as an experimental variable and measuring whether convolutional neural networks rely on background cues as spurious features.

Without binary masks, controlled counterfactual manipulation would not be possible.

---

### Reproducibility

This step is fully reproducible using the script:

```
scripts/create_binary_masks.py
```

The script deterministically converts trimaps into binary masks without randomness.

---

## Foreground Extraction

### Objective

Using the binary masks created in the previous step, foreground objects (pets) were extracted from each image while removing background pixels.

This produces images containing only the pet, with background pixels set to zero.

---

### Method

Foreground extraction was performed using element-wise masking.

Given:

* Image: I(x, y)
* Binary mask: M(x, y), where M = 1 indicates foreground and M = 0 indicates background

The extracted foreground image was computed as:

I_fg(x, y) = I(x, y) × M(x, y)

This operation preserves pet pixels while removing background pixels.

Background pixels were set to black (RGB = 0,0,0).

All extracted foreground images were saved in:

```
data/processed/foreground/
```

Total foreground images created: 7390

---

### Reason for This Step

Foreground extraction enables independent manipulation of background and object regions.

This is essential for constructing controlled datasets where:

* Background can be replaced with artificial colors
* Foreground remains unchanged
* Spurious correlations between background and class label can be introduced or reversed

This controlled separation allows causal testing of whether neural networks rely on background cues.

---

### Dataset Integrity

All 7390 images were successfully processed.

Minor JPEG decoding warnings were observed due to encoding irregularities in the original dataset, but these did not affect extraction correctness.

No images were skipped or lost.

---

### Reproducibility

This step is fully reproducible using:

```
scripts/extract_foreground.py
```

The process is deterministic and does not involve randomness.

---

## Construction of Dataset B — Background-Biased Dataset

### Objective

Dataset B was constructed to intentionally introduce a spurious correlation between background color and class label.

This enables controlled measurement of whether convolutional neural networks rely on background cues instead of foreground object features.

---

### Method

Foreground-extracted images were combined with artificially generated uniform color backgrounds.

Background colors were assigned deterministically based on class label:

* Cat images → green background (RGB: 0, 255, 0)
* Dog images → red background (RGB: 255, 0, 0)

Foreground pixels remained unchanged.

Background pixels were replaced using the binary mask:

Result(x, y) =

* Foreground(x, y), if mask(x, y) = 1
* Background_color, if mask(x, y) = 0

This ensures perfect correlation between background color and class label.

---

### Purpose in Experimental Design

This dataset creates a condition where background color is a highly predictive feature.

During training, neural networks may learn to rely on background color instead of animal features.

This enables testing whether the model learns:

* True object features (robust learning), or
* Spurious background correlations (biased learning)

---

### Dataset Properties

Total images: 7390

Background assignment:

* All cat images → green background
* All dog images → red background

No randomness was used in background assignment.

Dataset saved at:

```
data/datasets/dataset_B_biased/images/
```

---

### Role in Research

Dataset B serves as the biased training dataset in experiments.

Its purpose is to induce spurious feature reliance, which will later be tested using counterfactual evaluation.

---

### Reproducibility

Dataset B was generated using:

```
scripts/create_dataset_B_biased.py
```

The process is fully deterministic and reproducible.

---

## Construction of Dataset C — Counterfactual Dataset

### Objective

Dataset C was constructed to serve as a counterfactual evaluation set, where the correlation between background color and class label is intentionally reversed relative to Dataset B.

This enables direct testing of whether trained models rely on background cues.

---

### Method

Dataset C was created using the same foreground extraction and binary masks as Dataset B.

However, background color assignment was reversed:

* Cat images → red background (RGB: 255, 0, 0)
* Dog images → green background (RGB: 0, 255, 0)

Foreground pixels remained unchanged.

Background pixels were replaced deterministically using the binary masks.

This ensures that background color no longer aligns with the class label.

---

### Purpose in Experimental Design

Dataset C serves as a counterfactual test set.

If a model trained on Dataset B has learned true object features, it should maintain high accuracy on Dataset C.

If the model has learned background color as a spurious feature, accuracy will significantly decrease.

This performance difference provides quantitative evidence of background bias.

---

### Dataset Properties

Total images: 7390

Background assignment:

* All cat images → red background
* All dog images → green background

Dataset saved at:

```
data/datasets/dataset_C_counterfactual/images/
```

---

### Role in Research

Dataset C is used exclusively for evaluation.

It enables measurement of model robustness to background distribution shift.

This is the primary dataset used to test the research hypothesis.

---

### Reproducibility

Dataset C was generated using:

```
scripts/create_dataset_C_counterfactual.py
```

The process is fully deterministic and reproducible.

---

## Construction of Dataset D — Random Background Dataset

### Objective

Dataset D was constructed to introduce random background colors to the foreground-extracted images. This enables testing whether convolutional neural networks (CNNs) rely on background features when the backgrounds are random and unrelated to the object class.

---

### Method

Foreground-extracted images from the Oxford-IIIT Pet Dataset (processed in earlier steps) were combined with randomly generated background colors.

For each image:

1. The foreground (pet) image was retrieved from the `data/processed/foreground/` directory.
2. A binary mask (foreground vs. background) was used to separate the pet from the background.
3. A random background color was generated for each image. The background color was assigned randomly without any relation to the class label (cat or dog).
4. The foreground and background were merged using the binary mask to ensure the pet remains intact, while the background is replaced with the random color.

The result was saved as a new image in:

```
data/datasets/dataset_D_random/images/
```

Additionally, the images were structured into two directories for each class (cat and dog), following an ImageFolder format, and stored in:

```
data/datasets_structured/dataset_D/
```

---

### Purpose in Experimental Design

This dataset is crucial for testing the model's reliance on random background information. Since the background colors are random and have no connection to the object classes, it forces the neural network to focus on foreground features, assuming the model has not learned to use the background as a spurious feature.

This dataset will be used to compare the model's performance with those trained on biased datasets (Dataset B) and counterfactual datasets (Dataset C), helping evaluate whether CNNs can focus on true object features or are influenced by irrelevant background cues.

---

### Dataset Properties

- **Total images**: 7390
- **Image background**: Randomly generated, with no class-based correlation.
- **Class distribution**: As in the original dataset, split into:
  - `data/datasets_structured/dataset_D/cat/`
  - `data/datasets_structured/dataset_D/dog/`

The images are stored in a structured format compatible with ImageFolder-based training.

---

### Reproducibility

Dataset D was generated using the following script:

```
scripts/create_dataset_D_random.py
```

The process is deterministic and reproducible, ensuring consistent background generation and image structuring.