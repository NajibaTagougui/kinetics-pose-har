# Kinetics-400 Pose-Based Human Action Recognition for Ambient Intelligence


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
## 🙏 Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) team for the pose estimation library
- Kinetics-400 dataset creators (DeepMind)
