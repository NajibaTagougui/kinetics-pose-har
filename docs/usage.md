# Usage Guide

## Complete Pipeline (End-to-End Reproduction)

The following steps reproduce all results reported in the paper.

---

### Step 1 — Organise Raw Videos

```
data/raw/videos/
├── archery/
│   ├── video_001.mp4
│   └── video_002.mp4
├── clean and jerk/
│   └── ...
├── crawling baby/
│   └── ...
└── brushing teeth/
    └── ...
```

---

### Step 2 — Extract Pose Landmarks

```bash
python src/extract_poses.py \
    --input_dir  data/raw/videos \
    --output_dir data/processed/poses \
    --stride     15 \
    --max_frames 100
```

Expected output: one `.parquet` file per video in `data/processed/poses/`.

---

### Step 3 — Preprocess and Split

```bash
# 4-class subset (paper main benchmark)
python src/preprocess.py \
    --poses_dir  data/processed/poses \
    --output_dir data/processed \
    --benchmark  4class

# 8-class subset
python src/preprocess.py \
    --poses_dir  data/processed/poses \
    --output_dir data/processed \
    --benchmark  8class
```

This produces `train_4class.parquet`, `test_4class.parquet`, etc. using
**video-level splitting** to prevent temporal data leakage.

---

### Step 4 — Train Models

```bash
python src/train_model.py --config experiments/config_4class.yaml --output results/models/
python src/train_model.py --config experiments/config_8class.yaml --output results/models/
```

Expected console output (4-class):
```
INFO  Training complete in 13.2 s.
INFO  Accuracy: 0.7655  |  F1: 0.7677  |  Improvement: +0.5155
```

---

### Step 5 — Evaluate and Generate Figures

```bash
# Reproduce all paper figures at once
python src/evaluate.py --all

# Or evaluate a single model
python src/evaluate.py \
    --model     results/models/rf_4class_benchmark.pkl \
    --test_data data/processed/test_4class.parquet \
    --output    results/evaluation/
```

Output files:
- `confusion_4class_benchmark.png` — publication-ready confusion matrix (300 DPI)
- `per_class_4class_benchmark.png` — per-class accuracy bar chart
- `per_class_4class_benchmark.csv` — numeric results table
- `summary_4class_benchmark.json` — full metrics summary

---

## Using Pre-trained Models

```python
import joblib
import pandas as pd

# Load model
model = joblib.load("results/models/rf_4class_benchmark.pkl")

# Load new pose data (132 features per frame)
df = pd.read_parquet("data/processed/test_4class.parquet")
feat_cols = [c for c in df.columns if c not in {"video_id", "frame_index", "activity"}]
X = df[feat_cols].values

# Predict
predictions = model.predict(X)
probabilities = model.predict_proba(X)  # shape (N, 4)
```

---

## Using the Python API

```python
from src.preprocess import load_poses, filter_classes, split_by_video, get_feature_columns
from src.train_model import train_random_forest, evaluate_model
from src.utils import plot_feature_importance

# Load and prepare data
df = load_poses("data/processed/poses")
df = filter_classes(df, ["archery", "clean and jerk", "crawling baby", "brushing teeth"])
train_df, test_df = split_by_video(df, test_size=0.2)

feat_cols = get_feature_columns(train_df)
X_train = train_df[feat_cols].values
y_train = train_df["activity"].values
X_test  = test_df[feat_cols].values
y_test  = test_df["activity"].values

# Train
model = train_random_forest(X_train, y_train, {"n_estimators": 100})

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy : {metrics['accuracy']*100:.2f}%")
print(f"F1-Score : {metrics['f1_score']*100:.2f}%")

# Feature importance
plot_feature_importance(model.feature_importances_, top_k=15,
                        save_path="results/figures/feature_importance.png")
```

---

## Configuration Reference

All experiment parameters are controlled via YAML files in `experiments/`.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Experiment identifier | required |
| `data_path` | Directory containing train/test Parquets | required |
| `class_names` | List of activity classes (`null` = all) | required |
| `model_params.n_estimators` | Number of trees | 100 |
| `model_params.max_depth` | Tree depth (`null` = unlimited) | null |
| `training.test_size` | Fraction of videos for test | 0.20 |
| `training.random_state` | Random seed | 42 |
| `evaluation.figures_dpi` | Output figure resolution | 300 |
