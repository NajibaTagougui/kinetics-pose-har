"""
extract_poses.py
----------------
Extract 33 MediaPipe 3D pose landmarks from Kinetics-400 video files and
persist the results as Parquet files (one file per video).

Each output row contains 132 features:
    [x1, y1, z1, v1, x2, y2, z2, v2, ..., x33, y33, z33, v33]
plus metadata columns: video_id, frame_index, activity.

Usage
-----
    python src/extract_poses.py \\
        --input_dir  data/raw/videos \\
        --output_dir data/processed/poses \\
        --stride     15 \\
        --max_frames 100

Requirements
------------
    pip install opencv-python mediapipe pandas pyarrow tqdm
"""

import argparse
import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── MediaPipe initialisation ─────────────────────────────────────────────────
_mp_pose = mp.solutions.pose

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Build flat column names: nose_x, nose_y, nose_z, nose_v, left_eye_inner_x, ...
FEATURE_COLUMNS = [
    f"{name}_{axis}"
    for name in LANDMARK_NAMES
    for axis in ("x", "y", "z", "v")
]


# ─── Core extraction functions ────────────────────────────────────────────────

def extract_landmarks_from_frame(
    frame: np.ndarray,
    pose_model: mp.solutions.pose.Pose,
) -> list[float] | None:
    """
    Extract 132 pose features from a single BGR frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image array (H × W × 3).
    pose_model : mp.solutions.pose.Pose
        Initialised MediaPipe Pose instance.

    Returns
    -------
    list[float] or None
        Flattened [x, y, z, visibility] × 33, or None if no pose detected.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)

    if not results.pose_landmarks:
        return None

    return [
        val
        for lm in results.pose_landmarks.landmark
        for val in (lm.x, lm.y, lm.z, lm.visibility)
    ]


def process_video(
    video_path: Path,
    pose_model: mp.solutions.pose.Pose,
    stride: int = 15,
    max_frames: int = 100,
) -> pd.DataFrame:
    """
    Sample frames from a video and extract pose landmarks.

    Parameters
    ----------
    video_path : Path
        Path to the .mp4 (or compatible) video file.
    pose_model : mp.solutions.pose.Pose
        Shared MediaPipe Pose instance.
    stride : int
        Temporal sampling stride (process every Nth frame).
    max_frames : int
        Hard cap on the number of frames to extract per video.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns FEATURE_COLUMNS + [video_id, frame_index].
        Empty DataFrame if no landmarks were detected.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("Cannot open %s — skipping.", video_path)
        return pd.DataFrame()

    rows: list[list] = []
    frame_idx = 0
    extracted = 0

    while cap.isOpened() and extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            landmarks = extract_landmarks_from_frame(frame, pose_model)
            if landmarks is not None:
                rows.append(landmarks + [video_path.stem, frame_idx])
                extracted += 1

        frame_idx += 1

    cap.release()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["video_id", "frame_index"])


# ─── CLI entry point ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract MediaPipe 3D pose landmarks from Kinetics-400 videos."
    )
    p.add_argument("--input_dir",  type=str, required=True,
                   help="Root directory containing class sub-folders with .mp4 files.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory where per-video Parquet files will be written.")
    p.add_argument("--stride",     type=int, default=15,
                   help="Temporal sampling stride (default: 15).")
    p.add_argument("--max_frames", type=int, default=100,
                   help="Maximum frames extracted per video (default: 100).")
    p.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2],
                   help="MediaPipe model complexity (0=fastest, 2=most accurate).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path  = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_files = sorted(input_path.rglob("*.mp4"))
    if not video_files:
        log.error("No .mp4 files found under %s", input_path)
        return

    log.info("Found %d videos. stride=%d  max_frames=%d",
             len(video_files), args.stride, args.max_frames)

    with _mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose_model:

        total_frames = 0
        for vp in tqdm(video_files, desc="Extracting poses"):
            # Infer activity label from parent folder name
            activity = vp.parent.name

            df = process_video(vp, pose_model, args.stride, args.max_frames)
            if df.empty:
                log.debug("No landmarks detected in %s", vp.name)
                continue

            df["activity"] = activity
            out_file = output_path / f"{vp.stem}.parquet"
            df.to_parquet(out_file, index=False)
            total_frames += len(df)

    log.info("Done. Extracted %d frames → %s", total_frames, output_path)


if __name__ == "__main__":
    main()
