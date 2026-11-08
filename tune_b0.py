"""Dò ngưỡng B0 cải tiến trên tập TRAIN (không đụng val/test), lưu vào models/b0_rule.json.
Chạy:  python tune_b0.py --data data/windows.npz
       python tune_b0.py --data data/datasetv1.npz --name b0_rule_v1   # bộ test chung v1 (17 khung)
"""
import argparse
import itertools
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np

from b0_rule import B0Params, extract_all, predict_from_features
from data_utils import compute_metrics, load_dataset, save_model_json, video_level

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/windows.npz")
ap.add_argument("--name", default="b0_rule", help="lưu models/<name>.json (vd b0_rule_v1 cho datasetv1)")
args = ap.parse_args()

data = load_dataset(args.data)
Xtr, ytr, vtr = data["train"]

grid = itertools.product(
    (50.0, 55.0, 60.0),             # angle_th
    (0.1, 0.2, 0.3),                # shoulder_low_th
    (0.16, 0.18, 0.20, 0.22),       # v_fall_th (vận tốc rơi)
    (30.0, 35.0, 40.0),             # angle_change_th
    (0.35, 0.40, 0.45),             # occ_drop_th (độ sụt giảm mask khi khuất)
)

F_cache = {}
best = None

for a, h, v_th, c, occ in grid:
    # Trích xuất đặc trưng theo đúng occ_drop_th đang xét (dùng cache để tối ưu tốc độ)
    if occ not in F_cache:
        F_cache[occ] = extract_all(Xtr, B0Params(occ_drop_th=occ))
    F = F_cache[occ]

    p = B0Params(
        angle_th=a,
        shoulder_low_th=h,
        v_fall_th=v_th,
        angle_change_th=c,
        occ_drop_th=occ,
    )
    pred = predict_from_features(F, p)
    names, yt, yp = video_level(vtr, ytr, pred)
    vm = compute_metrics(yt, yp)
    wm = compute_metrics(ytr, pred)
    
    # Tối ưu hóa F1 video kết hợp F1 cửa sổ, ưu tiên cao video F1
    score = vm["f1"] * 0.7 + wm["f1"] * 0.3
    if best is None or score > best[0]:
        best = (score, vm, wm, p)

score, vm, wm, best_p = best
print(f"Tốt nhất trên TRAIN:")
print(f"  Điểm tổng hợp: {score:.3f}")
print(f"  Video  F1={vm['f1']:.3f} (P={vm['precision']:.3f}, R={vm['recall']:.3f}, TP={vm['tp']}, FP={vm['fp']}, TN={vm['tn']})")
print(f"  Window F1={wm['f1']:.3f} (P={wm['precision']:.3f}, R={wm['recall']:.3f})")
print(f"  Tham số: {best_p}")

metrics = {
    "train_score": score,
    "train_video_f1": vm["f1"],
    "train_window_f1": wm["f1"],
    "train_video_precision": vm["precision"],
    "train_video_recall": vm["recall"],
}
metrics["data"] = args.data
path = save_model_json(asdict(best_p), args.name, metrics)
print("Đã lưu", path)