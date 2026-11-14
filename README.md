# 🛰️ TLSF-CD: Temporal Localization and SAM3-Guided Label Fusion for Semi-Supervised Building Change Detection

TLSF-CD is a semi-supervised remote sensing change detection framework designed to improve pseudo-label quality when pixel-level annotations are limited. It combines temporal localization guidance with SAM3-guided label fusion to detect building changes in bi-temporal images.

## 📖 Overview

Semi-supervised change detection learns from a small set of labeled image pairs and a larger set of unlabeled pairs. Its effectiveness depends on the quality of pseudo-labels, particularly around small buildings, ambiguous boundaries, and cluttered backgrounds.

TLSF-CD addresses these challenges through two complementary components:

- **Temporal Localization Guidance (TLG)** predicts change-related object regions in each temporal image. Its auxiliary features help determine the temporal location of changed objects and guide the change detection branch to suppress background interference.
- **Adaptive SAM3-Guided Label Fusion** combines teacher predictions with object-level masks from SAM3. SAM3 supplies structural and boundary cues, while the teacher's change predictions help reject unsuitable SAM3 regions. The contribution of SAM3 guidance decreases progressively during training.

### Overall Architecture

<!-- Upload the final figure and replace this comment with:
![Overall architecture of TLSF-CD](assets/overall_architecture.png)
--><img width="1689" height="878" alt="image" src="https://github.com/user-attachments/assets/0f478e06-8d68-47d0-96ab-9b61b5f110b8" />


## 🧠 Model Configuration

TLSF-CD follows a teacher–student framework. The student uses a shared, pretrained DINOv2-Base encoder for the two temporal images. An exponential moving average (EMA) copy of the student serves as the teacher and generates pseudo-labels for unlabeled samples.

| Component | Configuration and role |
| --- | --- |
| Encoder | DINOv2-Base, fine-tuned during training |
| Encoder features | Intermediate features from layers 2, 5, 8, and 11 |
| Prediction heads | Three DPT-based heads: one change detection head and two temporal localization heads for T1 and T2 |
| DPT dimensions | Feature width of 128; projection widths of 96, 192, 384, and 768 |
| Temporal guidance | T1/T2 localization features guide spatial recalibration of change features |
| Teacher | EMA-updated copy of the student; maximum EMA coefficient of 0.996 |

### SAM 3 Configuration

TLSF-CD uses the frozen [SAM 3 image model](https://github.com/facebookresearch/sam3) to provide segmentation priors. Text-prompted masks are generated for the two temporal images separately. When a teacher-predicted change instance is not adequately matched by a text-prompted SAM3 mask, a foreground point and bounding box derived from that instance are used for complementary segmentation.

For each change instance, the best matching SAM3 mask may be a single instance or the union of overlapping SAM3 instances. A SAM3 mask is accepted when its IoU with the teacher-predicted change instance is at least 0.7; otherwise, the teacher instance is retained. Selected regions are aggregated to form a change-prior mask for subsequent label fusion.

| Setting | Value |
| --- | --- |
| SAM3 checkpoint | `sam3.pt`; obtained through the [official SAM 3 repository](https://github.com/facebookresearch/sam3) |
| Model state | Frozen and used in evaluation mode |
| Text prompts | `house`, `building`, and `roof` |
| Text-mask threshold | 0.5 |
| Text-prompt processor confidence threshold | 0.5 |
| Complementary prompts | A foreground point and bounding box for unmatched change instances |
| Instance-matching IoU threshold | 0.7 |

## 🗺️ Datasets

We evaluate TLSF-CD on three public high-resolution building change detection datasets. The original images are divided into non-overlapping 256 × 256 patches; the counts below refer to the patches used in our experiments.

| Dataset | Main change type | Resolution | Training | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: |
| [LEVIR-CD](https://github.com/justchenhao/LEVIR) | Building construction and demolition | 0.5 m | 7,120 | 1,024 | 2,048 |
| [WHU-CD](https://gpcv.whu.edu.cn/data/building_dataset.html) | Post-earthquake building changes | 0.075 m | 5,947 | 743 | 744 |
| [GZ-CD](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery) | Urban building changes across seasons | 0.55 m | 2,754 | 345 | 344 |

LEVIR-CD contains 637 original bi-temporal image pairs. WHU-CD covers Christchurch, New Zealand, before and after the 2012–2016 period. GZ-CD contains Google Earth imagery acquired between 2006 and 2019 under varying seasonal conditions.

Experiments use different labeled proportions to assess performance under limited supervision. Cross-dataset experiments additionally pair labeled target-domain samples with unlabeled samples from another dataset.

## ⚙️ Training Details

The framework is implemented in PyTorch and trained on one NVIDIA GeForce RTX 3090 GPU with 24 GB of memory.

| Setting | Value |
| --- | --- |
| Batch size | 4 |
| Training duration | 100 epochs |
| Optimizer | AdamW; weight decay of 0.01 |
| Initial learning rates | 5 × 10⁻⁶ for the encoder; 2 × 10⁻⁴ for newly introduced parameters |
| Learning rate schedule | Polynomial decay |
| Teacher pseudo-label confidence threshold | 0.95 |
| Weak augmentation | Random scaling, cropping, and horizontal flipping |
| Strong augmentation | CutMix, color jittering, grayscale conversion, and Gaussian blur |

The 256 × 256 image patches are processed at 252 × 252 for the change detection model, padded to 256 × 256 for SAM3 segmentation, and returned to 252 × 252 for spatial alignment with model predictions. Text-prompted SAM3 masks are generated separately from the original image patches.

The supervised branch learns from labeled change masks. For unlabeled samples, the teacher predicts on weakly augmented images, and the refined pseudo-labels supervise two strongly augmented student views. The temporal localization heads provide auxiliary supervision alongside the change detection objective.

## 📊 Representative Results on LEVIR-CD

The paper reports comparisons on LEVIR-CD, WHU-CD, and GZ-CD, as well as cross-dataset evaluations. This README presents representative quantitative and qualitative results on LEVIR-CD.

| Labeled training data | F1 (%) | IoU (%) |
| --- | ---: | ---: |
| 5% | 91.21 | 83.84 |
| 10% | 91.64 | 84.57 |
| 20% | 91.79 | 84.82 |
| 40% | 91.84 | 84.92 |

TLSF-CD obtains the highest F1 and IoU among the compared methods at each labeled ratio. With only 5% labeled training data, it achieves **91.21% F1** and **83.84% IoU**.

### Visual Comparison

The LEVIR-CD examples include low-contrast dense buildings and large, structurally complex changes. Under limited supervision, TLSF-CD produces more complete changed regions and clearer building boundaries than the compared methods.The figure below presents a visual comparison of different SSCD methods on the LEVIR-CD dataset.

<!-- Upload the final visual comparison and replace this comment with:
![Visual comparison of different SSCD methods on LEVIR-CD](assets/levir_visual_comparison.png)
--><img width="1348" height="939" alt="image" src="https://github.com/user-attachments/assets/147fdbb1-5577-4cb6-a4fb-0b6ae9db7dad" />

## 📢 Code Availability

Dataset-specific configuration files are available for [LEVIR-CD](configs/levir.yaml), [WHU-CD](configs/whu.yaml), and [GZ-CD](configs/gz.yaml). The complete training code, trained TLSF-CD checkpoints, and reproduction instructions will be released upon paper acceptance.

## 📝 Citation

Citation information will be added when the paper's publication details are available.
