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



## 🧠 Model Configuration

TLSF-CD adopts a teacher–student framework. The student uses a pretrained DINOv2-Base encoder to extract features from the two temporal images. Features from layers 2, 5, 8, and 11 are processed by three DPT-based heads: a change detection head and two temporal localization heads for T1 and T2. The encoder is fine-tuned during training. In the current model configuration, the DPT feature width is 128 and the four projection widths are 96, 192, 384, and 768.

The temporal localization heads predict the presence of change-related objects in each temporal image. Their semantic features guide a spatial gating module that enhances change-sensitive regions while suppressing background interference. An exponential moving average (EMA) copy of the student serves as the teacher for pseudo-label generation.

A frozen SAM3 model provides object-level segmentation priors. The text prompts “Building”, “House”, and “Roof” are applied to each temporal image independently. Unmatched change objects are further processed using automatically generated point and box prompts. A SAM3 mask is accepted when its overlap with the corresponding teacher-predicted instance reaches the IoU threshold of 0.7; otherwise, the teacher prediction is retained.


## 🗺️ Datasets

We evaluate TLSF-CD on three public high-resolution change detection datasets. All images are divided into non-overlapping 256 × 256 patches for the experiments.

| Dataset | Main change type | Spatial resolution | Training | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: |
| LEVIR-CD | Building changes | 0.5 m | 7,120 | 1,024 | 2,048 |
| WHU-CD | Post-earthquake building changes | 0.075 m | 5,947 | 743 | 744 |
| GZ-CD | Urban building changes across seasons | 0.55 m | 2755 | 345 | 345 |

LEVIR-CD contains 637 original bi-temporal image pairs. WHU-CD comprises aerial imagery of Christchurch, New Zealand, acquired in 2012 and 2016. GZ-CD is based on Google Earth imagery acquired between 2006 and 2019.

Experiments consider different proportions of labeled training samples to assess performance under limited supervision. Cross-dataset experiments additionally use labeled samples from the target dataset and unlabeled samples from another dataset.

## ⚙️ Training Details

| Setting | Value |
| --- | --- |
| Framework | PyTorch |
| Hardware | One NVIDIA GeForce RTX 3090 GPU with 24 GB memory |
| Input preparation | Non-overlapping 256 × 256 patches resized to 252 × 252 for DINOv2; padded to a multiple of 16 for SAM3 and aligned back to 252 × 252 |
| Batch size | 4 |
| Training epochs | 100 |
| Optimizer | AdamW |
| AdamW coefficients | β₁ = 0.9, β₂ = 0.999 |
| Weight decay | 0.01 |
| Initial encoder learning rate | 5 × 10⁻⁶ |
| Initial decoder learning rate | 2 × 10⁻⁴ |
| Learning rate schedule | Polynomial decay |
| Pseudo-label confidence threshold | 0.95 |
| SAM instance-matching IoU threshold | 0.7 |
| Maximum EMA coefficient | 0.996 |

Weak augmentation includes random cropping, scaling, and horizontal flipping. Strong augmentation includes CutMix, color jittering, grayscale conversion, and Gaussian blur. The teacher generates pseudo-labels from weakly augmented unlabeled images, which supervise two strongly augmented views of the student.

The supervised objective is a cross-entropy loss on labeled change masks. The unsupervised objective uses confidence-filtered pseudo-labels, while the two temporal localization heads are trained with an auxiliary combination of cross-entropy and Dice loss.

SAM3 masks and teacher predictions are progressively fused using

```math
\lambda(t)=\exp\left(-\frac{t}{t_{\max}}\right).
```

The unsupervised weight is

```math
\lambda_u(t)=1-\lambda(t),
```

giving the overall objective

```math
\mathcal{L}=\mathcal{L}_l+\lambda_u(t)\left(\mathcal{L}_u+\mathcal{L}_{\mathrm{aux}}\right).
```


## 📊 Representative Results on LEVIR-CD

The paper evaluates TLSF-CD on LEVIR-CD, WHU-CD, and GZ-CD. This README highlights the quantitative and qualitative results on LEVIR-CD; the complete experimental comparisons are reported in the manuscript.

### 📊 Quantitative Results

We conduct experiments for different semi-supervised change detection methods using 5%, 10%, 20%, and 40% labeled training samples of LEVIR-CD dataset. TLSF-CD achieves the highest F1 score and IoU at every labeled ratio. With only 5% labeled data, it reaches **91.21% F1** and **83.84% IoU**. 




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
