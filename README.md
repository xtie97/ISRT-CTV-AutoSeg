# Multi-Modality Artificial Intelligence for Involved-Site Radiation Therapy: Clinical Target Volume Delineation in High-Risk Pediatric Hodgkin Lymphoma

## 📌 Overview

This repository provides a deep learning framework for **automatic CTV delineation in ISRT**, with a focus on
oncologic imaging workflows in radiation therapy. The project aims to reduce contouring variability,
improve efficiency, and support clinically meaningful target definition.

CTV delineation for ISRT is clinically challenging and time-consuming, requiring radiation oncologists to integrate multi-time-point PET/CT imaging (baseline and interim) with post-chemotherapy planning CT. This repository implements and evaluates longitudinally aware, multi-modality deep learning models that automate this process and generate clinically useful CTVs with performance comparable to expert physicians.

This work is developed and validated using data from the Children’s Oncology Group (COG) AHOD1331 phase III clinical trial, representing one of the largest multi-institutional pediatric lymphoma imaging cohorts studied for this task

---

## Key Features

First deep learning framework for automated ISRT CTV delineation in pediatric lymphoma

Longitudinal modeling integrating:

Planning CT

Baseline PET/CT (PET1)

Interim PET/CT (PET2)

Comparison of CNN-based and Transformer-based architectures

Systematic evaluation of early vs. late multi-modality fusion

External validation on 58 patients from 24 institutions

Inter-observer variability (IOV) benchmarking against board-certified radiation oncologists

Blinded clinical reader study demonstrating comparable contour quality to physician-generated CTVs

---

## 🧠 Model Architectures
The framework supports three state-of-the-art 3D segmentation backbones:

| Architecture     | Type                   | Notes                                                   |
| ---------------- | ---------------------- | ------------------------------------------------------- |
| **SegResNet**    | CNN                    | Residual encoder-decoder                                |
| **ResUNet**      | CNN                    | nnU-Net–style residual U-Net                            |
| **SwinUNETR-V2** | Transformer-CNN hybrid | Window-based self-attention with stagewise convolutions |

Multi-Modality Fusion Strategies

Early Fusion
Concatenates planning CT + PET1 + PET2 at the input level.

Late Fusion
Processes CT and PET/CT through separate encoders and fuses learned features via:

Channel concatenation (CNNs)

Cross-attention (Transformers)

The late-fusion SwinUNETR model achieved the best overall performance on the external test cohort

🗂 Repository Structure
```bash
ISRT-CTV-AutoSeg/
├── configs/                # Training and inference configuration files
├── data/                   # Dataset structure templates and preprocessing
├── models/                 # SegResNet, ResUNet, SwinUNETR implementations
├── training/               # Training scripts (Auto3DSeg-based)
├── inference/              # Sliding-window inference and evaluation
├── utils/                  # Metrics, I/O, logging utilities
├── requirements.txt        # Python dependencies
└── README.md
```

