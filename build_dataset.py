"""
build_dataset.py
----------------
Gộp mọi file output/npz/*.npz thành tập huấn luyện cho ST-GCN.

Mỗi video: make_windows (cắt cửa sổ + gán nhãn) -> normalize_batch (chuẩn hóa).
Chia train/val/test THEO SUBJECT (cả người đó nằm trọn một phần) để không bị
rò rỉ dữ liệu giữa train và test.

Quy ước video_id (do main.py tạo): "<Subject>_<Loại>_<Số>", ví dụ
"Subject 1_Fall_01" hoặc "Subject 1_ADL_03". Subject là phần trước dấu "_" đầu
tiên; video là ngã nếu phần giữa chứa "fall" (không phân biệt hoa thường).

Nhãn: nếu có file khoảng ngã (CSV: video_id,start_sec,end_sec) thì cửa sổ chỉ
được nhãn 1 khi chồng lên khoảng đó. Video Fall không có khoảng nào bị BỎ QUA
(fall_web đổi mặc định; bản code_1 gán nhãn 1 cho TOÀN BỘ cửa sổ, làm model học
"nằm = ngã"). Muốn cách cũ: build(..., missing_interval="whole").

Chạy:  python build_dataset.py
Ra:    output/dataset.npz
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.features import normalize_batch
from src.skeleton_io import make_windows, seconds_to_frames


def subject_of(video_id: str) -> str:
    return video_id.split("_")[0]


def is_fall_video(video_id: str) -> bool:
    return any(p.lower() == "fall" for p in video_id.split("_")[1:-1])


def load_intervals(csv_path):
    """CSV (video_id,start_sec,end_sec) -> {video_id: [(start_sec, end_sec), ...]}."""
    out = defaultdict(list)
    if csv_path and Path(csv_path).exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                out[r["video_id"].strip().lower()].append(
                    (float(r["start_sec"]), float(r["end_sec"])))
    return out


def split_subjects(subjects, val_frac=0.15, test_frac=0.15, seed=0):
    subjects = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(subjects)))
    n = len(subjects)
    n_test = max(1, round(n * test_frac)) if n >= 3 else 0
    n_val = max(1, round(n * val_frac)) if n >= 3 else 0
    test = {subjects[i] for i in order[:n_test]}
    val = {subjects[i] for i in order[n_test:n_test + n_val]}
    train = set(subjects) - test - val
    return train, val, test


def build(npz_dir="output/npz", intervals_csv=None, out_path="output/dataset.npz",
          win_sec=2.0, stride_sec=0.5, min_overlap=0.3, min_valid_ratio=0.5,
          val_frac=0.15, test_frac=0.15, seed=0, missing_interval="skip"):
    intervals = load_intervals(intervals_csv)
    Xs, ys, vids, subs = [], [], [], []

    for f in sorted(Path(npz_dir).glob("*.npz")):
        vid = f.stem
        d = dict(np.load(f))
        d["fps"] = float(d["fps"])
        d["video_id"] = vid
        fall = is_fall_video(vid)

        fi = None
        if fall:
            secs = intervals.get(vid.lower())      # không phân biệt hoa/thường (Fall/FALL)
            if secs:
                fi = [(seconds_to_frames(a, d["fps"]), seconds_to_frames(b, d["fps"]))
                      for a, b in secs]
            elif missing_interval == "whole":
                print(f"[cảnh báo] {vid}: không có khoảng ngã, gán nhãn 1 cho cả video")
                fi = [(0, len(d["seq"]))]
            else:
                # fall_web: mặc định BỎ QUA. Gán nhãn 1 cho cả video làm cả đoạn đứng lẫn đoạn nằm yên thành "ngã",
                # khiến model báo ngã khi người chỉ nằm bình thường. Thêm dòng vào fall_intervals.csv rồi chạy lại.
                print(f"[bỏ qua] {vid}: video ngã chưa có khoảng ngã trong file nhãn")
                continue

        X, M, y, g = make_windows(d, win_sec, stride_sec, fall_intervals=fi,
                                  min_overlap=min_overlap, min_valid_ratio=min_valid_ratio)
        if len(X) == 0:
            print(f"[bỏ qua] {vid}: không có cửa sổ nào đủ điều kiện")
            continue
        Xs.append(normalize_batch(X, M, float(d["aspect"])))
        ys.append(y)
        vids += [vid] * len(y)
        subs += [subject_of(vid)] * len(y)

    if not Xs:
        raise SystemExit(f"Không có dữ liệu trong {npz_dir}")
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    vids = np.array(vids)
    subs = np.array(subs)

    train, val, test = split_subjects(subs, val_frac, test_frac, seed)
    if not val or not test:
        print("[cảnh báo] ít hơn 3 subject: toàn bộ dữ liệu vào train, chưa có val/test")
    part = np.where(np.isin(subs, list(test)), "test",
                    np.where(np.isin(subs, list(val)), "val", "train"))

    pack = {}
    for name in ("train", "val", "test"):
        m = part == name
        pack[f"X_{name}"], pack[f"y_{name}"] = X[m], y[m]
        pack[f"video_{name}"] = vids[m]
        print(f"{name:5s}: {int(m.sum()):5d} cửa sổ | ngã: {int(y[m].sum())}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **pack)
    print(f"Đã lưu: {out_path} | X: {X.shape}")
    return pack


if __name__ == "__main__":
    build(intervals_csv="data/fall_intervals.csv")   # đổi/bỏ nếu chưa có file nhãn