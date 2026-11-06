"""Đánh giá B0 trên train/val/test (mức cửa sổ và mức video), liệt kê video sai.
Chạy:  python eval_b0.py --data output/dataset.npz"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
from data_utils import (compute_metrics, load_dataset, load_model_json, save_model_json, video_level)
from b0_rule import B0Params, B0Rule

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/window.npz")
ap.add_argument("--model", default="b0_rule", help="tên file trong models/ (bỏ .json); không có thì dùng mặc định")
args = ap.parse_args()

data = load_dataset(args.data)
try:
    model = B0Rule.from_params(load_model_json(args.model)["params"]); print("Dùng tham số từ models/%s.json" % args.model)
except FileNotFoundError:
    model = B0Rule(B0Params()); print("Chưa có models/%s.json, dùng ngưỡng mặc định" % args.model)
print("Tham số:", model.get_params(), "\n")

report = {}
for split in ("train", "val", "test"):
    X, y, v = data[split]
    pred = model.predict(X)
    w = compute_metrics(y, pred)
    names, yt, yp = video_level(v, y, pred)
    vm = compute_metrics(yt, yp)
    report[split] = dict(window=w, video=vm)
    print(f"[{split}] cửa sổ: P={w['precision']:.3f} R={w['recall']:.3f} F1={w['f1']:.3f} Acc={w['accuracy']:.3f}"
          f" | TP={w['tp']} FP={w['fp']} FN={w['fn']} TN={w['tn']}")
    print(f"[{split}] video  : P={vm['precision']:.3f} R={vm['recall']:.3f} F1={vm['f1']:.3f}"
          f" | TP={vm['tp']} FP={vm['fp']} FN={vm['fn']} TN={vm['tn']}")
    if split in ("val", "test"):
        miss = [n for n, a, b in zip(names, yt, yp) if a == 1 and b == 0]
        false_alarm = [n for n, a, b in zip(names, yt, yp) if a == 0 and b == 1]
        print(f"    video ngã bị BỎ SÓT : {miss}")
        print(f"    video thường báo NHẦM: {false_alarm}")

Path("output").mkdir(exist_ok=True)
Path("output/b0_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
print("\nĐã lưu output/b0_report.json")
