# API Reference

---

## `src.extract_poses`

### `extract_landmarks_from_frame(frame, pose_model)`

Extract 132 pose features from a single BGR video frame.

**Parameters**
- `frame` (`np.ndarray`) — BGR image array (H × W × 3)
- `pose_model` (`mp.solutions.pose.Pose`) — initialised MediaPipe Pose instance

**Returns** `list[float] | None`  
Flattened `[x, y, z, visibility] × 33`, or `None` if no pose is detected.

---

### `process_video(video_path, pose_model, stride=15, max_frames=100)`

Sample frames from a video and extract pose landmarks.

**Parameters**
- `video_path` (`Path`) — path to `.mp4` file
- `pose_model` — shared MediaPipe Pose instance
- `stride` (`int`) — temporal sampling stride (process every Nth frame)
- `max_frames` (`int`) — hard cap on extracted frames

**Returns** `pd.DataFrame`  
Columns: `FEATURE_COLUMNS + ["video_id", "frame_index"]`. Empty if no landmarks found.

---

## `src.preprocess`

### `load_poses(poses_dir)`

Concatenate all per-video Parquet files into a single DataFrame.

**Parameters**
- `poses_dir` (`str | Path`) — directory containing `*.parquet` files

**Returns** `pd.DataFrame`

---

### `filter_classes(df, classes)`

Keep only rows whose `activity` column is in `classes`.

**Parameters**
- `df` (`pd.DataFrame`)
- `classes` (`list[str] | None`) — `None` keeps all classes

**Returns** `pd.DataFrame`

---

### `split_by_video(df, test_size=0.2, random_state=42)`

Video-level stratified train/test split — no temporal leakage.

**Parameters**
- `df` (`pd.DataFrame`)
- `test_size` (`float`) — fraction of videos for test set
- `random_state` (`int`) — reproducibility seed

**Returns** `tuple[pd.DataFrame, pd.DataFrame]` — `(train_df, test_df)`

---

### `get_feature_columns(df)`

Return the 132 pose feature column names, excluding metadata.

**Returns** `list[str]`

---

## `src.train_model`

### `train_random_forest(X_train, y_train, params, random_state=42)`

Fit a scikit-learn `RandomForestClassifier`.

**Parameters**
- `X_train` (`np.ndarray`, shape `(N, 132)`)
- `y_train` (`np.ndarray`, shape `(N,)`)
- `params` (`dict`) — keys: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- `random_state` (`int`)

**Returns** `RandomForestClassifier`

---

### `evaluate_model(model, X_test, y_test)`

Compute accuracy, weighted F1, improvement over chance, and classification report.

**Returns** `dict` with keys:
```
accuracy    float   Overall accuracy
f1_score    float   Weighted F1-score
chance      float   1/n_classes
improvement float   accuracy − chance
n_classes   int
n_test      int
report      str     Full sklearn classification_report
```

---

## `src.evaluate`

### `run_evaluation(model_path, test_path, output_dir, name=None)`

Full evaluation pipeline: loads model + test set, computes metrics, saves confusion matrix PNG, per-class bar chart, CSV, and JSON summary.

**Parameters**
- `model_path` (`Path`) — `.pkl` model file
- `test_path` (`Path`) — Parquet test file
- `output_dir` (`Path`) — directory for all output files
- `name` (`str | None`) — tag used in output filenames

**Returns** `dict` — same schema as `evaluate_model` + `per_class` nested dict

---

## `src.utils`

### `plot_feature_importance(importances, top_k=15, save_path=None, dpi=300)`

Horizontal bar chart of the top-k Gini feature importances.

**Parameters**
- `importances` (`np.ndarray`, shape `(132,)`)
- `top_k` (`int`)
- `save_path` (`Path | None`)
- `dpi` (`int`)

**Returns** `pd.DataFrame` — top-k features sorted by importance

---

### `dataset_summary(df)`

Print frame count, unique videos, classes, and top-10 classes by frame count.

---

### `frames_per_video_histogram(df, save_path=None, dpi=300)`

Histogram of frame counts per video with mean/median lines.

---

## Constants

| Name | Module | Value |
|------|--------|-------|
| `FEATURE_COLUMNS` | `extract_poses`, `utils` | `list[str]` of 132 column names |
| `LANDMARK_NAMES` | `extract_poses`, `utils` | `list[str]` of 33 MediaPipe landmark names |
| `CLASSES_4` | `preprocess` | 4-class benchmark activity list |
| `CLASSES_8` | `preprocess` | 8-class benchmark activity list |
