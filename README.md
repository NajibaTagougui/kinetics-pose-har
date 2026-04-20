# Kinetics-400 Pose-Based Human Action Recognition for Ambient Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A privacy-preserving framework for human action recognition using 3D skeletal pose landmarks instead of RGB video. This repository contains the complete code and data pipeline for the paper:

> **Privacy-Preserving Human Action Recognition for Ambient Intelligence: A 3D Pose-Based Benchmark**  
> *Najiba Tagougui and Monji Kherallah*  
> Springer Nature Journal

## 📋 Overview

This framework replaces intrusive RGB monitoring with 3D skeletal pose landmarks extracted using MediaPipe Pose Landmarker. The system achieves:

| Benchmark     | Accuracy    | Improvement over Chance  |
|---------------|-------------|--------------------------|
| 4-Class Subset | **76.55%** | +51.55% (p < 0.001)     |
| 8-Class Subset | **43.87%** | +31.40% (p < 0.001)     |

### Key Features

- ✅ **Privacy-by-Design**: No RGB storage or transmission — raw pixels are purged immediately after pose extraction
- ✅ **Edge-Deployable**: Lightweight Random Forest (~8 MB), millisecond inference per frame
- ✅ **Interpretable**: Gini feature importance exposes which body joints drive recognition
- ✅ **Large-Scale**: 4,163,828 frames · 328 action classes · 8,370 videos
- ✅ **Reproducible**: Complete code, configs, and processed data available

## 📊 Dataset

The processed dataset contains:

| Property           | Value             |
|--------------------|-------------------|
| Total frames       | 4,163,828         |
| Total videos       | 8,370             |
| Action classes     | 328               |
| Features per frame | 132 (33 × [x,y,z,ν]) |
| File format        | Parquet (~1.27 GB)|
| Avg. landmark visibility | 74.7%      |

### Evaluation Subsets

| Subset   | Classes | Frames    | Videos | Train Frames | Test Frames |
|----------|---------|-----------|--------|--------------|-------------|
| Full     | 328     | 4,163,828 | 8,370  | —            | —           |
| 8-Class  | 8       | 366,683   | 290    | 294,931      | 71,752      |
| 4-Class  | 4       | 183,093   | 168    | 142,872      | 40,221      |

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10 or higher required
python --version

# Clone repository
git clone https://github.com/NajibaTagougui/kinetics-pose-har.git
cd kinetics-pose-har

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Extract Poses from Videos

```bash
python src/extract_poses.py \
    --input_dir data/raw/videos \
    --output_dir data/processed/poses \
    --stride 15 \
    --max_frames 100
```

### Train Model

```bash
# Train on 4-class subset
python src/train_model.py \
    --config experiments/config_4class.yaml \
    --output results/models/

# Train on 8-class subset
python src/train_model.py \
    --config experiments/config_8class.yaml \
    --output results/models/
```

### Evaluate Model

```bash
python src/evaluate.py \
    --model results/models/rf_4class_benchmark.pkl \
    --test_data data/processed/test_4class.parquet \
    --output results/evaluation/
```

## 📈 Results

### 4-Class Benchmark (Best Subset)

| Class          | Accuracy | Test Samples | Ambient Context        |
|----------------|----------|--------------|------------------------|
| Archery        | 84.85%   | 8,811        | Precision upper-body   |
| Clean and Jerk | 82.94%   | 12,580       | Athletic power monitoring |
| Crawling Baby  | 79.21%   | 7,070        | Pediatric safety       |
| Brushing Teeth | 61.90%   | 11,760       | Daily living activity  |
| **Overall**    | **76.55%** | **40,221** | F1-Score: 76.77%       |

### 8-Class Benchmark (Full Diversity)

| Class               | Accuracy | Test Samples |
|---------------------|----------|--------------|
| Clean and Jerk      | 68.27%   | 7,696        |
| Crawling Baby       | 64.29%   | 3,920        |
| Brushing Teeth      | 63.07%   | 14,080       |
| Archery             | 40.91%   | 5,874        |
| Blowing Out Candles | 38.14%   | 9,794        |
| Arm Wrestling       | 34.61%   | 10,438       |
| Blowing Glass       | 26.32%   | 7,980        |
| Air Drumming        | 24.81%   | 11,970       |
| **Overall**         | **43.87%** | **71,752** | 

### Model Comparison (4-Class)

| Model                  | Accuracy   | F1-Score   | Gain over Chance |
|------------------------|------------|------------|------------------|
| **Random Forest (Ours)** | **76.55%** | **76.77%** | **+51.55%**    |
| SVM (RBF Kernel)       | 65.00%     | 65.90%     | +40.00%          |
| MLP (Deep Learning)    | 14.70%     | 14.60%     | −10.30%          |

## 🏗️ System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Perception     │     │  Anonymization  │     │  Intelligence   │
│  Layer          │────▶│  Gateway        │────▶│  Layer          │
│  (RGB Frames)   │     │  (MediaPipe)    │     │  (Random Forest)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   Zero-Persistence        132D Pose Vector        76.55% Accuracy
   (No storage)            (x, y, z, ν × 33)       Edge Deployable
```

## 📁 Repository Structure

```
kinetics-pose-har/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
├── setup.py                   # Installation script
├── .gitignore
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── extract_poses.py       # MediaPipe pose extraction
│   ├── preprocess.py          # Data preprocessing & splitting
│   ├── train_model.py         # Random Forest training
│   ├── evaluate.py            # Evaluation & confusion matrix
│   └── utils.py               # Plotting utilities
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── experiments/               # YAML configuration files
│   ├── config_4class.yaml
│   ├── config_8class.yaml
│   └── config_full.yaml
│
├── results/                   # Output directory
│   ├── figures/
│   ├── models/
│   └── logs/
│
├── tests/                     # Unit tests
│   ├── test_extraction.py
│   ├── test_preprocessing.py
│   └── test_model.py
│
└── docs/                      # Detailed documentation
    ├── installation.md
    ├── usage.md
    ├── api_reference.md
    └── contributing.md
```

## 🧪 Reproducibility

To reproduce all paper results from scratch:

```bash
# 1. Extract poses from raw Kinetics-400 videos
python src/extract_poses.py \
    --input_dir data/raw/videos \
    --output_dir data/processed/poses \
    --stride 15 --max_frames 100

# 2. Preprocess and split (video-level, no leakage)
python src/preprocess.py --split_strategy video_level

# 3. Train both benchmarks
python src/train_model.py --config experiments/config_4class.yaml
python src/train_model.py --config experiments/config_8class.yaml

# 4. Generate evaluation reports and confusion matrices
python src/evaluate.py --all

# 5. Reproduce paper figures
jupyter nbconvert --to notebook --execute notebooks/03_results_visualization.ipynb
```

Expected outputs:

| Step            | Output file           | Expected result        |
|-----------------|-----------------------|------------------------|
| Pose Extraction | full_dataset.parquet  | 4,163,828 frames, 1.27 GB |
| 4-Class Train   | rf_4class.pkl         | 76.55% accuracy        |
| 8-Class Train   | rf_8class.pkl         | 43.87% accuracy        |
| Figures         | results/figures/      | 300 DPI publication-ready PNGs |

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api_reference.md)
- [Contributing Guidelines](docs/contributing.md)

## 🔬 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{tagougui2024privacy,
  title     = {Privacy-Preserving Human Action Recognition for Ambient Intelligence: A 3D Pose-Based Benchmark},
  author    = {Tagougui, Najiba and Kherallah, Monji},
  journal   = {Springer Nature Journal},
  year      = {2024},
  url       = {https://github.com/NajibaTagougui/kinetics-pose-har}
}
```

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please see [docs/contributing.md](docs/contributing.md) for guidelines.

## 📧 Contact

- **Corresponding author**: Najiba Tagougui — najiba.tagougui@isims.usf.tn
- **Issues**: [GitHub Issues](https://github.com/NajibaTagougui/kinetics-pose-har/issues)

## 🙏 Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) team for the pose estimation library
- Kinetics-400 dataset creators (DeepMind)
- Springer Nature for publication support

## ⚠️ Disclaimer

This code is provided for research purposes. The privacy-preserving nature of the pipeline has been validated against GDPR principles, but users should ensure compliance with local data protection regulations.
