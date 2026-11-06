#!/usr/bin/env python3
"""
evaluate.py
-----------
So sánh các checkpoint (ST-GCN, LSTM, ...) trên CÙNG tập test, đủ số liệu cho báo cáo B1:
  - theo cửa sổ: Accuracy, Precision, Recall, F1, Specificity, AUC ở ngưỡng 0.5 và ngưỡng lưu trong checkpoint,
  - theo video: video ngã có bị bỏ sót không, video sinh hoạt có bị báo nhầm không,
  - số tham số, dung lượng file, thời gian suy luận một cửa sổ (batch 1) trên thiết bị đang chạy.

Ví dụ (từ thư mục gốc project):
  python evaluate.py models/best_stgcn.pt models/best_lstm.pt
  python evaluate.py models/*.pt --split val --csv output/compare.csv
  python evaluate.py models/best_stgcn.pt models/best_lstm.pt --html output/compare_test.html
      -> báo cáo HTML một file (biểu đồ, ROC, ma trận nhầm lẫn, từng video, khoảng tin cậy) để đưa vào báo cáo.
         Cùng nội dung với trang "So sánh model" của web (http://127.0.0.1:5000/compare).
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_utils import load_splits, metrics_at, roc_auc, video_level  # noqa: E402


def resolve_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() or p.exists() else ROOT / p


def latency_ms(clf, X, runs=200, warmup=20) -> float:
    """Thời gian trung bình (ms) cho MỘT cửa sổ — giống lúc chạy thật mỗi 5 khung một lần."""
    one = X[:1]
    for _ in range(warmup):
        clf.predict_batch(one)
    t0 = time.perf_counter()
    for _ in range(runs):
        clf.predict_batch(one)
    return (time.perf_counter() - t0) * 1000.0 / runs


def evaluate(paths, data_path, split="test", device=None, runs=200):
    from src.classifier import FallClassifier
    data = load_splits(resolve_path(data_path))
    X, y, vid = data[split]
    rows = []
    for p in paths:
        p = resolve_path(p)
        clf = FallClassifier(p, device=device, verbose=False)
        if clf.window_size != X.shape[1]:
            print(f"[cảnh báo] {p.name}: checkpoint cửa sổ {clf.window_size} khung, dataset {X.shape[1]} khung")
        prob = clf.predict_batch(X)
        thr = clf.fall_threshold
        m05, mt = metrics_at(y, prob, 0.5), metrics_at(y, prob, thr)
        mv = video_level(y, prob, vid, thr)
        rows.append(dict(
            model=p.name, arch=clf.arch, params=clf.num_params,
            size_mb=round(p.stat().st_size / 1e6, 2),
            latency_ms=round(latency_ms(clf, X, runs=runs), 2) if runs > 0 else None,
            auc=None if roc_auc(y, prob) is None else round(roc_auc(y, prob), 4),
            acc=round(m05["acc"], 4), prec=round(m05["prec"], 4), rec=round(m05["rec"], 4),
            f1=round(m05["f1"], 4), spec=round(m05["spec"], 4),
            tp=m05["tp"], fn=m05["fn"], fp=m05["fp"], tn=m05["tn"],
            thr=round(thr, 2), f1_thr=round(mt["f1"], 4), rec_thr=round(mt["rec"], 4),
            prec_thr=round(mt["prec"], 4),
            video_rec=round(mv["rec"], 4), video_spec=round(mv["spec"], 4),
            video_fn=mv["fn"], video_fp=mv["fp"], videos=mv["videos"],
            device=str(clf.device),
        ))
    return data, rows


def print_table(rows, split, n, n_fall):
    print(f"\nTập {split}: {n} cửa sổ ({n_fall} ngã)\n")
    hdr = ["model", "arch", "params", "MB", "ms/cửa sổ", "AUC", "Acc", "P", "R", "F1", "Spec",
           "ngưỡng", "F1@ngưỡng", "R video", "Spec video"]
    body = [[r["model"], r["arch"], f"{r['params']:,}", r["size_mb"], r["latency_ms"], r["auc"], r["acc"],
             r["prec"], r["rec"], r["f1"], r["spec"], r["thr"], r["f1_thr"], r["video_rec"], r["video_spec"]]
            for r in rows]
    widths = [max(len(str(x)) for x in col) for col in zip(hdr, *body)]
    line = lambda cells: "  ".join(str(c).rjust(w) for c, w in zip(cells, widths))  # noqa: E731
    print(line(hdr))
    print("  ".join("-" * w for w in widths))
    for b in body:
        print(line(b))
    print("\nP, R, F1, Spec: ngưỡng 0.5, theo cửa sổ. 'R video' = tỉ lệ video ngã được phát hiện;"
          " 'Spec video' = tỉ lệ video sinh hoạt không bị báo nhầm (ở ngưỡng của checkpoint).")
    for r in rows:
        print(f"  {r['model']}: TP={r['tp']} FN={r['fn']} FP={r['fp']} TN={r['tn']} | "
              f"video bỏ sót {r['video_fn']}, báo nhầm {r['video_fp']} / {r['videos']} video")


def main(argv=None):
    ap = argparse.ArgumentParser(description="So sánh checkpoint phát hiện ngã",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default=None, help="cpu | cuda (đo tốc độ nên dùng cpu cho giống thiết bị biên)")
    ap.add_argument("--runs", type=int, default=200, help="số lần đo tốc độ (0 = bỏ qua)")
    ap.add_argument("--csv", default=None, help="ghi bảng kết quả ra CSV")
    ap.add_argument("--html", default=None, help="ghi báo cáo so sánh HTML một file (mở bằng trình duyệt)")
    args = ap.parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("Chưa cài PyTorch: pip install torch")
    # PowerShell / cmd không tự mở rộng "models/lstm_*.pt" -> tự làm ở đây
    paths = []
    for c in args.checkpoints:
        if any(ch in c for ch in "*?["):
            hits = sorted(glob.glob(c)) or sorted(glob.glob(str(ROOT / c)))
            if not hits:
                sys.exit(f"Không có file nào khớp {c}")
            paths += hits
        else:
            paths.append(c)
    data, rows = evaluate(paths, args.data, args.split, args.device, args.runs)
    _, y, _ = data[args.split]
    print_table(rows, args.split, len(y), int(y.sum()))
    if args.html:
        from web.compare import compare_models, render_report
        res = compare_models(paths, args.data, args.split, device=args.device or "cpu", runs=args.runs)
        out = Path(args.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_report(res, standalone=True), encoding="utf-8")
        print("\nKết luận:")
        for v in res["verdict"]:
            print("  -", v)
        print(f"Đã ghi báo cáo {out}")
    if args.csv:
        out = resolve_path(args.csv) if Path(args.csv).is_absolute() else Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nĐã ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
