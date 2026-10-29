"""Tạo dataset cửa sổ cho huấn luyện (ST-GCN, LSTM, ...) từ NPZ đã trích + nhãn trong metadata.csv.

    python run_web.py build-dataset          ->  output/<cấu hình>/dataset/windows.npz

Quy trình:
  metadata.csv (label, fall_start_s, fall_end_s, subject_id, split)
  + output/<cấu hình>/npz/<video_id>.npz (skeleton_io.from_poses)
  -> skeleton_io.make_windows  (cửa sổ trượt theo thời gian, nhãn theo độ chồng lấn với đoạn ngã)
  -> features.normalize_batch  (cùng hàm chuẩn hóa với lúc chạy thật trong FallClassifier)

windows.npz chứa:
  X       [N, W, 17, 3] float32  đã chuẩn hóa (x, y, mask) — to_model_input(X) -> [N, 3, W, 17]
  y       [N] int64              0 = không ngã, 1 = ngã
  video   [N] str                video_id của từng cửa sổ (chia train/val THEO VIDEO, không theo cửa sổ)
  subject [N] str                subject_id (rỗng nếu chưa điền) — chia theo người để tránh rò rỉ
  split   [N] str                train / val / test / "" (lấy từ metadata)
  win_sec, stride_sec, fps, window_size  — window_size phải trùng classifier.window_size khi chạy thật
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import normalize_batch
from src.skeleton_io import make_windows, seconds_to_frames

from .engine import load_npz, npz_dir, output_dir
from .config import resolve
from .videos import load_metadata

DEFAULTS = dict(win_sec=2.0, stride_sec=0.5, min_overlap=0.3, min_valid_ratio=0.0)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _str(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else str(v)


def build_dataset(cfg: dict, out_path: str | Path | None = None) -> tuple[Path, dict, list[str]]:
    """Trả (đường dẫn windows.npz, thống kê, danh sách cảnh báo)."""
    dcfg = {**DEFAULTS, **(cfg.get("dataset") or {})}
    meta = load_metadata(resolve(cfg["paths"]["metadata"]))
    ndir = npz_dir(cfg)
    notes: list[str] = []
    Xs, ys, vids, subs, splits = [], [], [], [], []
    fps_seen: set[float] = set()
    W = None

    for r in meta.itertuples():
        vid = str(r.video_id)
        p = ndir / f"{vid}.npz"
        if not p.exists():
            notes.append(f"{vid}: chưa trích skeleton (thiếu {p.name}) — bỏ qua")
            continue
        label = _str(r.label)
        intervals = []
        if label == "fall":
            a, b = _num(r.fall_start_s), _num(r.fall_end_s)
            if a is None or b is None or b <= a:
                notes.append(f"{vid}: video ngã thiếu/sai fall_start_s, fall_end_s — bỏ qua")
                continue
        item = load_npz(p)
        item["video_id"] = vid
        fps = round(item["fps"], 3)
        fps_seen.add(fps)
        if label == "fall":
            intervals = [(seconds_to_frames(a, item["fps"]), seconds_to_frames(b, item["fps"]))]
        elif label != "adl":
            notes.append(f"{vid}: nhãn '{label}' không phải fall/adl — bỏ qua")
            continue
        X, M, y, _ = make_windows(item, win_sec=dcfg["win_sec"], stride_sec=dcfg["stride_sec"],
                                  fall_intervals=intervals, min_overlap=dcfg["min_overlap"],
                                  min_valid_ratio=dcfg["min_valid_ratio"])
        if not len(X):
            notes.append(f"{vid}: không có cửa sổ nào đạt min_valid_ratio — bỏ qua")
            continue
        if W is not None and X.shape[1] != W:
            notes.append(f"{vid}: fps {fps} cho cửa sổ {X.shape[1]} khung, khác {W} — bỏ qua "
                         "(trích lại mọi video cùng target_fps)")
            continue
        W = X.shape[1]
        if label == "fall" and not y.any():
            notes.append(f"{vid}: video ngã nhưng không cửa sổ nào được gán nhãn ngã "
                         "(đoạn ngã quá ngắn so với min_overlap?)")
        Xs.append(normalize_batch(X, M, aspect=item["aspect"]))
        ys.append(y)
        vids += [vid] * len(y)
        subs += [_str(r.subject_id)] * len(y)
        splits += [_str(r.split)] * len(y)

    if not Xs:
        raise ValueError("Không tạo được cửa sổ nào. Hãy trích skeleton và điền nhãn trong metadata.csv. "
                         + ("; ".join(notes[:5])))
    X, y = np.concatenate(Xs).astype(np.float32), np.concatenate(ys)
    fps = sorted(fps_seen)[0]
    out = Path(out_path) if out_path else output_dir(cfg) / "dataset" / "windows.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, video=np.array(vids), subject=np.array(subs),
                        split=np.array(splits), win_sec=dcfg["win_sec"], stride_sec=dcfg["stride_sec"],
                        fps=fps, window_size=W)
    by_split = pd.DataFrame({"split": splits, "y": y}).groupby("split")["y"].agg(["count", "sum"])
    stats = dict(windows=int(len(y)), fall=int(y.sum()), no_fall=int(len(y) - y.sum()),
                 videos=len(set(vids)), window_size=int(W), fps=fps,
                 by_split={(k or "(trống)"): dict(windows=int(v["count"]), fall=int(v["sum"]))
                           for k, v in by_split.iterrows()})
    return out, stats, notes


def load_dataset(path: str | Path) -> dict:
    """Đọc windows.npz -> dict (dùng trong script huấn luyện)."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    for k in ("win_sec", "stride_sec", "fps"):
        d[k] = float(d[k])
    d["window_size"] = int(d["window_size"])
    return d
