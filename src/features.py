"""
features.py
-----------
Chuẩn hóa clip cho ST-GCN. Dùng CHUNG cho huấn luyện (PC) và chạy thật (board).

Đầu vào:
  seq  [W,17,>=2]  x, y (0-1); ô không quan sát = 0 (kết quả của from_poses)
  mask [W,17]      True = quan sát thật
  aspect           rộng/cao của khung gốc (item["aspect"])
Đầu ra [W,17,3]: x, y đã chuẩn hóa, và kênh 3 = mask (0/1).
Không dùng conf, vì board MaixCAM không trả conf từng khớp.

Quy tắc: MỘT tâm và MỘT tỉ lệ cho cả clip (không tính riêng từng khung), để
không xóa mất chuyển động hạ thấp người khi ngã.
"""
from __future__ import annotations

import numpy as np

from .constants import BONES, FLIP_INDEX, HIPS


def normalize_clip(seq, mask, aspect: float = 1.0) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(seq.shape[:2] + (3,), dtype=np.float32)
    if not mask.any():                              # clip không thấy khớp nào
        return out

    xy = seq[..., :2].copy()
    xy[..., 0] *= aspect                            # sửa méo theo tỉ lệ khung hình

    # Tâm: trung bình hai hông (khi quan sát được), không có thì trung bình mọi khớp.
    hip_ok = mask[:, HIPS]
    if hip_ok.any():
        center = xy[:, HIPS][hip_ok].mean(axis=0)
    else:
        center = xy[mask].mean(axis=0)

    # Tỉ lệ: trung vị độ dài các "xương" quan sát được (giống SkeletonFilter).
    lengths = []
    for a, b in BONES:
        both = mask[:, a] & mask[:, b]
        if both.any():
            lengths.append(np.linalg.norm(xy[both, a] - xy[both, b], axis=-1))
    scale = float(np.median(np.concatenate(lengths))) if lengths else 0.0
    if scale < 1e-3:
        scale = 1.0

    xy = (xy - center) / scale
    xy[~mask] = 0.0
    out[..., :2] = xy
    out[..., 2] = mask
    return out


def normalize_batch(X, M, aspect: float = 1.0) -> np.ndarray:
    """X [N,W,17,3], M [N,W,17] (từ make_windows) -> [N,W,17,3] đã chuẩn hóa."""
    if len(X) == 0:
        return np.zeros((0,) + tuple(np.shape(X)[1:3]) + (3,), dtype=np.float32)
    return np.stack([normalize_clip(x, m, aspect) for x, m in zip(X, M)])


def flip_clip(clip) -> np.ndarray:
    """Lật ngang clip ĐÃ chuẩn hóa: đổi dấu x và hoán đổi trái/phải."""
    out = np.asarray(clip)[:, FLIP_INDEX].copy()
    out[..., 0] *= -1
    out[..., 0][out[..., 2] == 0] = 0.0             # giữ ô thiếu bằng 0 (tránh -0)
    return out