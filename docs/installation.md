# Installation Guide

## System Requirements

| Component | Minimum         | Recommended       |
|-----------|-----------------|-------------------|
| OS        | Linux / macOS / Windows (WSL2) | Ubuntu 22.04 LTS |
| Python    | 3.10            | 3.11              |
| RAM       | 8 GB            | 16 GB             |
| Storage   | 5 GB (processed data only) | 50 GB (raw videos + data) |
| GPU       | Not required    | NVIDIA CUDA (for pose extraction speed) |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/NajibaTagougui/kinetics-pose-har.git
cd kinetics-pose-har
```

---

## Step 2 — Create a Virtual Environment

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

---

## Step 3 — Install Dependencies

### Inference / training only (no video processing)
```bash
pip install -r requirements.txt
```

### Full installation — including pose extraction from raw videos
```bash
pip install -r requirements.txt
pip install ".[pose_extraction]"
# or manually:
pip install opencv-python mediapipe
```

### Development installation (includes testing tools)
```bash
pip install -e ".[dev]"
```

---

## Step 4 — Download Processed Data (optional)

Pre-extracted pose Parquet files are available on Zenodo:

```bash
# 4-class subset (~200 MB)
wget https://zenodo.org/record/XXXXXXX/files/train_4class.parquet
wget https://zenodo.org/record/XXXXXXX/files/test_4class.parquet

# 8-class subset (~500 MB)
wget https://zenodo.org/record/XXXXXXX/files/train_8class.parquet
wget https://zenodo.org/record/XXXXXXX/files/test_8class.parquet

# Full dataset (~1.27 GB)
wget https://zenodo.org/record/XXXXXXX/files/complete_pose_dataset.parquet
```

Place files in `data/processed/`.

---

## Step 5 — Verify Installation

```bash
python -c "import sklearn, pandas, numpy, seaborn; print('All dependencies OK')"
```

Run the unit test suite:
```bash
pytest tests/ -v
```

---

## Troubleshooting

### MediaPipe on Apple Silicon (M1/M2/M3)
```bash
pip install mediapipe-silicon
```

### Out-of-memory errors during training
Use the `--config` option to reduce `n_estimators`, or filter to a smaller subset:
```bash
python src/preprocess.py --benchmark 4class --min_visibility 0.5
```

### Parquet read errors
Ensure `pyarrow` is installed:
```bash
pip install pyarrow --upgrade
```
