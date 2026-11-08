#!/usr/bin/env python3
"""
build_lying_negatives.py
------------------------
datasetv1 gần như không có mẫu NẰM mà KHÔNG NGÃ (train: 2118 cửa sổ ADL, 32 cửa sổ nằm, 0 cửa sổ nằm yên), trong
khi 67% cửa sổ Fall là tư thế nằm, nên model học "nằm = ngã". Script này thêm các mẫu âm (nhãn 0) tạo từ chính
các cửa sổ Fall của CÙNG split (không rò rỉ subject giữa train/val/test):

  still  người giữ nguyên tư thế cuối của cú ngã (nằm yên), có rung nhẹ của skeleton
  slow   cú ngã bị kéo giãn thời gian thành động tác nằm xuống chậm, rồi nằm yên
         (giữ lại khi vai hạ < 1.6 thân/s và xoay < 70 độ/s, xa vùng tốc độ của cú ngã thật)

Một phần mẫu bị che cả nhóm khớp (hông / vai / chân), giống người nằm bị chăn, giường che.
Mẫu được chuẩn hóa lại bằng features.normalize_clip như dữ liệu thật. Cửa sổ gốc giữ nguyên.

Chạy:
    python build_lying_negatives.py                     # data/datasetv1.npz -> data/datasetv1_lying.npz
Train:
    python tune_b1.py --data data/datasetv1_lying.npz
Bộ test chung gốc data/testsets/v1_test KHÔNG đổi. Bộ test có thêm tình huống nằm:
    python test_tool.py --make-testset v1_lying --data data/datasetv1_lying.npz --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.features import normalize_clip  # noqa: E402
from src.motion import motion_features  # noqa: E402

MAX_SHOULDER, MAX_ROTATION = 1.6, 70.0     # trần tốc độ của mẫu "nằm xuống chậm"
GROUPS = ([11, 12], [5, 6], [13, 14, 15, 16])      # hông, vai, chân


def _finish(xy, mask, rng, occlude_p=0.2):
    """xy [T,17,2], mask [T,17] -> clip đã chuẩn hóa [T,17,3], có thể che một nhóm khớp."""
    mask = mask.copy()
    if rng.random() < occlude_p:
        mask[:, GROUPS[rng.integers(len(GROUPS))]] = False
    if mask.sum() < 5 * len(mask):
        return None
    seq = np.concatenate([xy, np.ones(xy.shape[:2] + (1,), np.float32)], axis=-1)
    return normalize_clip(seq, mask, aspect=1.0)


def _jitter(T, rng):
    """Rung skeleton lúc đứng yên: nhiễu từng khung + trôi chậm chung cả người."""
    noise = rng.normal(0.0, rng.uniform(0.01, 0.04), (T, 17, 2))
    drift = np.cumsum(rng.normal(0.0, 0.006, (T, 1, 2)), axis=0)
    return (noise + drift).astype(np.float32)


def make_still(clip, rng):
    """Giữ một trong các tư thế cuối của clip Fall suốt cửa sổ."""
    T = len(clip)
    i = T - 1 - int(rng.integers(0, 4))
    xy, m = clip[i, :, :2], clip[i, :, 2] > 0
    if m.sum() < 5:
        return None
    flick = rng.random((T, 17)) > 0.02                # thỉnh thoảng rớt khớp như YOLO thật
    return _finish(np.tile(xy, (T, 1, 1)) + _jitter(T, rng), np.tile(m, (T, 1)) & flick, rng)


def make_slow(clip, rng, tries=5):
    """Kéo giãn đoạn đổ người thành nằm xuống chậm, nối thêm đoạn nằm yên, cắt ngẫu nhiên một cửa sổ."""
    T = len(clip)
    xy, m = clip[..., :2], clip[..., 2] > 0
    ok = m[:, [5, 6, 11, 12]].all(axis=1)
    torso = xy[:, [5, 6]].mean(1) - xy[:, [11, 12]].mean(1)
    angle = np.degrees(np.arctan2(np.abs(torso[:, 0]), -torso[:, 1]))
    if ok.sum() < 3:
        p = T // 2
    else:
        rate = np.abs(np.diff(np.where(ok, angle, np.nan)))
        p = int(np.nanargmax(rate)) if np.isfinite(rate).any() else T // 2
    a, b = max(0, p - 5), min(T - 1, p + 4)
    for _ in range(tries):
        n = int(rng.integers(T + 1, 2 * T))
        pos = np.linspace(a, b, n)
        lo, hi = np.floor(pos).astype(int), np.ceil(pos).astype(int)
        w = (pos - lo)[:, None, None].astype(np.float32)
        fill = np.where(m[..., None], xy, np.nan)                 # nội suy chỉ giữa các khớp quan sát được
        moved = (1 - w) * fill[lo] + w * fill[hi]
        got = np.isfinite(moved).all(-1)
        long_xy = np.concatenate([np.nan_to_num(moved), np.tile(np.nan_to_num(moved[-1]), (T, 1, 1))]).astype(np.float32)
        long_m = np.concatenate([got, np.tile(got[-1], (T, 1))])
        s = int(rng.integers(0, len(long_xy) - T + 1))
        out = _finish(long_xy[s:s + T] + _jitter(T, rng), long_m[s:s + T], rng)
        if out is None:
            continue
        f = motion_features(out, 10.0)
        if f["shoulder_drop"] < MAX_SHOULDER and f["rotation"] < MAX_ROTATION:
            return out
    return None


def augment_split(X, y, video, rng, still_frac, slow_frac):
    """Trả (X, y, video) gồm dữ liệu gốc + mẫu âm tạo thêm, và số mẫu mỗi loại."""
    n_adl = int((y == 0).sum())
    fall = np.flatnonzero(y == 1)
    ang = motion_features(X[fall], 10.0)["angle_end"]
    weights = np.where(ang > 45, 0.7 / max(1, (ang > 45).sum()), 0.3 / max(1, (ang <= 45).sum()))
    weights = weights / weights.sum()
    new_x, new_v, counts = [], [], {}
    for kind, frac, fn in (("still", still_frac, make_still), ("slow", slow_frac, make_slow)):
        target, made = int(round(n_adl * frac)), 0
        for _ in range(target * 3):
            if made >= target:
                break
            i = int(rng.choice(fall, p=weights))
            out = fn(X[i], rng)
            if out is None:
                continue
            subject, _, tag = str(video[i]).split("_", 2)
            new_x.append(out)
            new_v.append(f"{subject}_ADL_{tag}-{kind}")
            made += 1
        counts[kind] = made
    Xn = np.concatenate([X, np.stack(new_x)]).astype(np.float32)
    yn = np.concatenate([y, np.zeros(len(new_x), np.int64)])
    vn = np.concatenate([video.astype(str), np.array(new_v)])
    return Xn, yn, vn, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Thêm mẫu âm 'nằm mà không ngã' vào dataset",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--data", default="data/datasetv1.npz")
    ap.add_argument("--out", default="data/datasetv1_lying.npz")
    ap.add_argument("--still", type=float, default=0.25, help="số mẫu nằm yên, tính theo tỉ lệ số cửa sổ ADL")
    ap.add_argument("--slow", type=float, default=0.15, help="số mẫu nằm xuống chậm, tính theo tỉ lệ số cửa sổ ADL")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    with np.load(ROOT / args.data, allow_pickle=False) as d:
        src = {k: d[k] for k in d.files}
    out = {k: v for k, v in src.items() if k.endswith("_subjects")}
    for split in ("train", "val", "test"):
        X, y, v, c = augment_split(src[f"X_{split}"], src[f"y_{split}"], src[f"video_{split}"], rng,
                                   args.still, args.slow)
        out[f"X_{split}"], out[f"y_{split}"], out[f"video_{split}"] = X, y, v
        f = motion_features(X[len(src[f"y_{split}"]):], 10.0)
        print(f"{split:5s}: gốc {len(src[f'y_{split}'])} -> {len(y)} cửa sổ | thêm {c} | vai hạ p95 "
              f"{np.percentile(f['shoulder_drop'], 95):.2f} thân/s, xoay p95 {np.percentile(f['rotation'], 95):.0f} độ/s")
    np.savez_compressed(ROOT / args.out, **out)
    print(f"Đã lưu {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
