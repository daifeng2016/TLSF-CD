# 🛰️ TLSF-CD: Temporal Localization and SAM3-Guided Label Fusion for Semi-Supervised Building Change Detection

TLSF-CD is a semi-supervised remote sensing change detection framework that combines temporal localization and SAM-guided label fusion to improve pseudo-label quality with limited pixel-level annotations.

## 📖 Overview

Reliable pseudo-labels are essential for learning from unlabeled bi-temporal images. However, ambiguous boundaries, small objects, and background interference can cause false alarms and missed changes.

TLSF-CD addresses these challenges by explicitly modeling the temporal location of changed objects and progressively refining pseudo-labels with object-level priors from SAM.

## 🧠 Method

TLSF-CD consists of two complementary components:

- **Temporal localization guidance:** An auxiliary segmentation branch captures spatial cues from both temporal images. Trend-aware supervision helps identify when changed objects appear, while semantic-aware gating suppresses background noise and false positives.
- **Adaptive SAM-guided label fusion:** SAM-derived object masks provide structural guidance during early training. As training progresses, network predictions contribute to dynamic correction while high-confidence changed regions are preserved.

The framework uses a DINOv2-Base encoder and a DPT-based decoder to jointly produce bi-temporal segmentation maps and change maps.

### Overall Architecture

<!-- Add the final architecture image to the repository, then replace this comment with:
![Overall architecture of TLSF-CD](figures/TLSF-CD_framework.png)
--><img width="1625" height="818" alt="image" src="https://github.com/user-attachments/assets/1a38e76e-879b-4bfc-b5e5-84732632809a" />




## 🗺️ Datasets

We evaluate TLSF-CD on three public high-resolution change detection datasets. All images are divided into non-overlapping 256 × 256 patches for the experiments.

| Dataset | Main change type | Spatial resolution | Training | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: |
| LEVIR-CD | Building changes | 0.5 m | 7,120 | 1,024 | 2,048 |
| WHU-CD | Post-earthquake building changes | 0.075 m | 5,947 | 743 | 744 |
| GZ-CD | Urban building changes across seasons | 0.55 m | 2755 | 345 | 345 |

LEVIR-CD contains 637 original bi-temporal image pairs. WHU-CD comprises aerial imagery of Christchurch, New Zealand, acquired in 2012 and 2016. GZ-CD is based on Google Earth imagery acquired between 2006 and 2019.

Experiments consider different proportions of labeled training samples to assess performance under limited supervision. Cross-dataset experiments additionally use labeled samples from the target dataset and unlabeled samples from another dataset.

## ⚙️ Experimental Setup

| Item | Setting |
| --- | --- |
| Framework | PyTorch |
| GPU | One NVIDIA GeForce RTX 3090, 24 GB |
| Encoder | DINOv2-Base |
| Decoder | DPT-based decoder |
| Input size | 256 × 256 patches resized to 252 × 252 for the network |
| SAM input | Edge-padded to a multiple of 16, then aligned with 252 × 252 predictions |
| SAM text prompts | “Building”, “House”, “Roof” |
| Batch size | 4 |
| Training epochs | 100 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Encoder learning rate | 5 × 10⁻⁶ |
| Decoder learning rate | 2 × 10⁻⁴ |
| Learning rate schedule | Polynomial decay |
| Instance-matching IoU threshold | 0.7 |
| Confidence threshold | 0.95 |

Weak augmentation includes random cropping, scaling, and horizontal flipping. Strong augmentation includes CutMix, color jittering, grayscale conversion, and Gaussian blur.

## 📊 Representative Results on LEVIR-CD

The paper evaluates TLSF-CD on LEVIR-CD, WHU-CD, and GZ-CD. This README highlights the quantitative and qualitative results on LEVIR-CD; the complete experimental comparisons are reported in the manuscript.

### 📊 Quantitative Results

Table I compares semi-supervised change detection methods using 5%, 10%, 20%, and 40% labeled training samples of LEVIR-CD dataset. TLSF-CD achieves the highest F1 score and IoU at every labeled ratio. With only 5% labeled data, it reaches **91.21% F1** and **83.84% IoU**. Its F1 score exceeds CBFFNet by **0.67 percentage points** and is **0.73 percentage points** below the fully supervised result.




### 🖼️ Visual Comparison

The visual comparison includes challenging cases such as low-contrast dense buildings and large-scale complex changes in LEVIR-CD dataset. Compared with the other methods, TLSF-CD better preserves complete changed regions and building boundaries, including when only 5% of the training samples are labeled.

<!-- Add the final visual comparison images, then insert them here:
![Qualitative results on LEVIR-CD](figures/LEVIR-CD_results.png)
![Qualitative results on WHU-CD](figures/WHU-CD_results.png)
![Qualitative results on GZ-CD](figures/GZ-CD_results.png)
--><img width="1037" height="839" alt="image" src="https://github.com/user-attachments/assets/2cd1dfff-25b9-45c0-8f17-729d224711ab" />



## 📢 Code Availability

The complete training and evaluation code, configuration files, pretrained weights, and reproduction instructions are planned for release upon paper acceptance.

## 📝 Citation

<!-- Add the final BibTeX entry when publication information is available. -->
