"""
data_utils.py  (Phúc)
---------------------
Khung chương trình nạp dữ liệu skeleton, lưu kết quả và quản lý model.
Dùng chung cho B0 (luật), B1 (LSTM), B2 (ST-GCN): cả nhóm cùng gọi load_dataset().

dataset.npz do build_dataset.py (Thành) tạo ra, gồm:
  X_train/val/test  [N, W, 17, 3]  x, y đã chuẩn hóa; kênh 3 = mask (1 = khớp quan sát thật)
  y_train/val/test  [N]            0 = không ngã, 1 = ngã
  video_train/...   [N]            video_id của từng cửa sổ, dạng "Subject 1_Fall_01"
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


def load_dataset(path="data/windows.npz"):
    """Trả dict {split: (X, y, video_ids)}. Đọc được cả dataset.npz (X_train...) lẫn windows.npz (X, split)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không thấy {path}. Đặt windows.npz vào thư mục data/.")
    d = np.load(path, allow_pickle=False)
    out = {}
    if "X_train" in d.files:                          # định dạng cũ
        for s in SPLITS:
            out[s] = (d[f"X_{s}"], d[f"y_{s}"], d[f"video_{s}"])
    elif {"X", "y", "split"} <= set(d.files):         # định dạng windows.npz
        split = d["split"].astype(str)
        for s in SPLITS:
            m = split == s
            out[s] = (d["X"][m], d["y"][m], d["video"][m])
    else:
        raise ValueError(f"{path}: không nhận ra định dạng (khóa: {sorted(d.files)})")
    for s, (X, y, v) in out.items():
        assert X.ndim == 4 and X.shape[2:] == (17, 3), f"X_{s} sai shape: {X.shape}"
        assert len(X) == len(y) == len(v), f"Độ dài X/y/video của {s} không khớp"
    describe(out)
    return out


def describe(data):
    print(f"{'tập':6s}{'cửa sổ':>8s}{'ngã':>7s}{'tỉ lệ ngã':>11s}{'video':>7s}  {'mask TB':>8s}")
    for s, (X, y, v) in data.items():
        rate = y.mean() if len(y) else 0
        print(f"{s:6s}{len(y):8d}{int(y.sum()):7d}{rate:11.1%}{len(set(v)):7d}  {X[..., 2].mean():8.2f}")


def save_model_json(params: dict, name: str, metrics: dict | None = None, models_dir="models"):
    """Lưu tham số model (B0 là luật nên chỉ là một dict ngưỡng) kèm số liệu, thời gian."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.json"
    payload = dict(name=name, saved_at=datetime.now().isoformat(timespec="seconds"),
                   params=params, metrics=metrics or {})
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return path


def load_model_json(name: str, models_dir="models"):
    path = Path(models_dir) / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compute_metrics(y_true, y_pred):
    """Precision / recall / F1 / accuracy / ma trận nhầm lẫn cho lớp ngã (1)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1,
                accuracy=(tp + tn) / max(len(y_true), 1),
                specificity=tn / (tn + fp) if tn + fp else 0.0,
                tp=tp, fp=fp, fn=fn, tn=tn)


def video_level(video_ids, y_true, y_pred):
    """Gộp theo video: video 'ngã' nếu có ÍT NHẤT một cửa sổ được dự đoán ngã.
    Đây là cách gần với ứng dụng thật (cảnh báo khi có một cửa sổ báo ngã)."""
    vids = np.asarray(video_ids)
    names = sorted(str(n) for n in set(vids))
    yt = np.array([int(y_true[vids == n].max()) for n in names])
    yp = np.array([int(y_pred[vids == n].max()) for n in names])
    return names, yt, yp
