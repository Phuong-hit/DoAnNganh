"""
skeleton_io.py
--------------
Chuyển list[PoseResult] -> mảng cho huấn luyện / suy luận, và cắt cửa sổ.

Quy ước (train và real-time dùng CHUNG hàm from_poses):
  seq   [T,17,3]  x, y (0-1), conf. Ô không tin cậy = 0 hết (x=y=conf=0).
  mask  [T,17]    True = quan sát THẬT (x, y hữu hạn và conf > 0).
                  Khớp "giữ tạm" của SkeletonFilter (conf = 0) là False.
  valid [T]       khung có người.
  t, frame_idx [T]; fps; aspect (rộng/cao khung gốc).

Huấn luyện nên đọc từ NPZ do main.py lưu, không đọc từ CSV (CSV mất aspect).
"""
from __future__ import annotations

import numpy as np

from .constants import KEYPOINT_NAMES, NUM_KEYPOINTS


def _build_mask(seq):
    """seq [T,17,3] (có thể chứa NaN) -> (seq sạch, mask). Ô mask=False về 0."""
    seq = np.asarray(seq, dtype=np.float32).copy()
    conf = np.nan_to_num(seq[:, :, 2], nan=0.0)
    mask = np.isfinite(seq[:, :, 0]) & np.isfinite(seq[:, :, 1]) & (conf > 0)
    seq = np.nan_to_num(seq, nan=0.0)
    seq[:, :, 2] = conf
    seq[~mask] = 0.0
    return seq, mask


def seconds_to_frames(seconds, fps):
    """Đổi mốc thời gian (giây) sang chỉ số khung SAU lấy mẫu."""
    return int(round(seconds * fps))


def load_sequence(csv_path):
    """CSV -> dict. CHỈ ĐỂ DEBUG: CSV không có aspect; huấn luyện hãy dùng NPZ."""
    import pandas as pd  # chỉ cần khi đọc CSV
    from pathlib import Path

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.sort_values("frame_idx").drop_duplicates("frame_idx", keep="last")

    full = np.arange(df["frame_idx"].iloc[0], df["frame_idx"].iloc[-1] + 1)
    df = df.set_index("frame_idx").reindex(full)

    T = len(df)
    raw = np.full((T, NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
    for i, name in enumerate(KEYPOINT_NAMES):
        raw[:, i, 0] = df[f"{name}_x"].to_numpy(dtype=np.float32)
        raw[:, i, 1] = df[f"{name}_y"].to_numpy(dtype=np.float32)
        raw[:, i, 2] = df[f"{name}_conf"].to_numpy(dtype=np.float32)
    seq, mask = _build_mask(raw)                    # cùng quy tắc với from_poses

    valid = df["valid"].fillna(0).to_numpy().astype(bool)
    dt = np.diff(df["t"].dropna().to_numpy())
    dt = dt[dt > 0]
    fps = float(1.0 / np.median(dt)) if len(dt) else 10.0

    return dict(seq=seq, mask=mask, valid=valid, frame_idx=full, fps=fps,
                video_id=Path(str(csv_path)).stem)


def from_poses(poses):
    """list[PoseResult] -> dict cho huấn luyện / suy luận (không cần pandas)."""
    if not poses:
        raise ValueError("poses rỗng.")
    raw = np.stack([np.asarray(p.joints, dtype=np.float32) for p in poses])
    seq, mask = _build_mask(raw)
    valid = np.array([bool(p.valid) for p in poses])
    t = np.array([p.t for p in poses], dtype=np.float64)
    frame_idx = np.array([p.frame_idx for p in poses], dtype=np.int64)

    dt = np.diff(t)
    dt = dt[dt > 0]
    fps = float(1.0 / np.median(dt)) if len(dt) else 10.0
    aspects = [p.aspect for p in poses if p.width and p.height]
    aspect = float(aspects[0]) if aspects else 1.0

    return dict(seq=seq, mask=mask, valid=valid, t=t,
                frame_idx=frame_idx, fps=fps, aspect=aspect)


def make_windows(item, win_sec=2.0, stride_sec=0.5, fall_intervals=None,
                 min_overlap=0.3, min_valid_ratio=0.0):
    """Cắt cửa sổ trượt theo THỜI GIAN.

    fall_intervals: list (start_frame, end_frame) theo chỉ số khung SAU lấy mẫu
                    (dùng seconds_to_frames nếu nhãn ghi theo giây).
    min_valid_ratio: bỏ cửa sổ có tỉ lệ khung valid < ngưỡng (0 = giữ hết).

    Trả (X [N,W,17,3], M [N,W,17], y [N], groups [N]).
    groups = video_id của từng cửa sổ, để chia train/val THEO VIDEO/SUBJECT.
    """
    fps = item["fps"]
    W = max(1, int(round(win_sec * fps)))
    S = max(1, int(round(stride_sec * fps)))
    seq, mask = item["seq"], item["mask"]
    valid = item.get("valid")
    if valid is None:
        valid = np.ones(len(seq), dtype=bool)
    T = len(seq)
    if T < W:                                   # pad cuối bằng 0, mask=False
        pad = W - T
        seq = np.pad(seq, ((0, pad), (0, 0), (0, 0)))
        mask = np.pad(mask, ((0, pad), (0, 0)))
        valid = np.pad(valid, (0, pad))
        T = W

    X, M, y = [], [], []
    for s in range(0, T - W + 1, S):
        if valid[s:s + W].mean() < min_valid_ratio:
            continue
        X.append(seq[s:s + W])
        M.append(mask[s:s + W])
        lab = 0
        for a, b in (fall_intervals or []):
            ov = max(0, min(s + W, b) - max(s, a))
            if ov / W >= min_overlap:
                lab = 1
        y.append(lab)

    if not X:
        return (np.zeros((0, W, NUM_KEYPOINTS, 3), np.float32),
                np.zeros((0, W, NUM_KEYPOINTS), bool),
                np.zeros(0, np.int64), np.zeros(0, dtype=object))
    groups = np.array([str(item.get("video_id", ""))] * len(X), dtype=object)
    return np.stack(X), np.stack(M), np.array(y, dtype=np.int64), groups