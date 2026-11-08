#!/usr/bin/env python3
"""
tune_b1.py
----------
Tinh chỉnh B1 (LSTM): huấn luyện nhiều cấu hình × nhiều seed trên CÙNG dataset, chọn cấu hình theo VAL
(không nhìn test), rồi chép checkpoint được chọn thành models/b1_lstm.pt để test_tool.py / web dùng.

Tiêu chí chọn (chỉ dùng val), quy tắc "một độ lệch chuẩn" (--rule 1se, mặc định):
    - cấu hình tốt nhất = F1 val trung bình qua các seed cao nhất (ổn định hơn một seed may mắn);
    - mọi cấu hình có F1 val trung bình >= (tốt nhất - độ lệch chuẩn của nó) coi là ngang nhau
      (val chỉ có 3 subject: chênh lệch nhỏ hơn mức dao động giữa các seed là nhiễu);
    - trong nhóm ngang nhau chọn model ÍT THAM SỐ nhất (nhẹ cho thiết bị biên, ít overfit).
    --rule best: lấy thẳng F1 val trung bình cao nhất (rồi AUC val, rồi ít tham số).
Trong cấu hình được chọn, lấy seed có F1 val cao nhất.

Chạy (từ gốc project):
    python tune_b1.py                                   # mọi cấu hình, 3 seed, data/datasetv1.npz
    python tune_b1.py --configs base bi small --seeds 1 # chạy nhanh
    python tune_b1.py --list                            # xem các cấu hình
    python tune_b1.py --reselect                        # chọn lại từ runs.csv, không train lại
Ra:
    models/b1/<cấu hình>_s<seed>.pt (+ .report.json, .history.csv)
    output/b1_tuning/runs.csv      (mỗi lần chạy một dòng)
    output/b1_tuning/configs.csv   (gộp theo cấu hình: trung bình ± độ lệch chuẩn)
    models/b1_lstm.pt              (checkpoint được chọn)
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
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

# tên -> (tham số train.py thêm vào, mô tả)
CONFIGS: dict[str, tuple[list[str], str]] = {
    "base":     ([], "mặc định: embed 128, hidden 128, 2 lớp, có vận tốc, dropout 0.3"),
    "bi":       (["--model-kw", "bidirectional=true"], "LSTM hai chiều"),
    "small":    (["--model-kw", "embed=64", "--model-kw", "hidden=64", "--model-kw", "layers=1"],
                 "nhỏ: embed 64, hidden 64, 1 lớp (nhẹ cho thiết bị biên)"),
    "wide":     (["--model-kw", "embed=192", "--model-kw", "hidden=256"], "rộng: embed 192, hidden 256"),
    "novel":    (["--model-kw", "velocity=false"], "bỏ đặc trưng vận tốc (kiểm tra vận tốc có giúp không)"),
    "drop5":    (["--model-kw", "dropout=0.5"], "dropout 0.5 (chống overfit khi ít subject)"),
    "slow":     (["--lr", "5e-4", "--clip-grad", "1.0", "--epochs", "80"], "lr 5e-4 + cắt gradient 1.0, 80 epoch"),
    "bi_drop5": (["--model-kw", "bidirectional=true", "--model-kw", "dropout=0.5", "--clip-grad", "1.0"],
                 "hai chiều + dropout 0.5 + cắt gradient"),
}


def run_one(name, extra, seed, data, out_dir, device, epochs, patience):
    import train as T
    out = out_dir / f"{name}_s{seed}.pt"
    argv = ["--arch", "lstm", "--data", data, "--out", str(out), "--seed", str(seed),
            "--epochs", str(epochs), "--patience", str(patience)] + list(extra)
    if device:
        argv += ["--device", device]
    args = T.build_parser().parse_args(argv)          # tham số sau ghi đè tham số trước (vd --epochs của "slow")
    rep = T.train(args)
    return dict(
        config=name, seed=seed, checkpoint=str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out),
        params=rep["params"], best_epoch=rep["best_epoch"], epochs_run=rep["epochs_run"],
        minutes=rep["train_minutes"], threshold=rep["threshold"],
        val_f1=round(rep["val_thr"]["f1"], 4), val_prec=round(rep["val_thr"]["prec"], 4),
        val_rec=round(rep["val_thr"]["rec"], 4), val_auc=round(rep["val_auc"] or 0.0, 4),
        test_f1=round(rep["test_thr"]["f1"], 4), test_prec=round(rep["test_thr"]["prec"], 4),
        test_rec=round(rep["test_thr"]["rec"], 4), test_auc=round(rep["test_auc"] or 0.0, 4),
        test_video_rec=round(rep["test_video"]["rec"], 4), test_video_spec=round(rep["test_video"]["spec"], 4),
    )


def summarize(runs):
    by = {}
    for r in runs:
        by.setdefault(r["config"], []).append(r)
    rows = []
    for name, rs in by.items():
        f = lambda k: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
        rows.append(dict(
            config=name, description=CONFIGS.get(name, ([], ""))[1], seeds=len(rs), params=rs[0]["params"],
            val_f1_mean=round(f("val_f1").mean(), 4), val_f1_std=round(f("val_f1").std(), 4),
            val_auc_mean=round(f("val_auc").mean(), 4),
            test_f1_mean=round(f("test_f1").mean(), 4), test_f1_std=round(f("test_f1").std(), 4),
            test_rec_mean=round(f("test_rec").mean(), 4), test_prec_mean=round(f("test_prec").mean(), 4),
            minutes_mean=round(f("minutes").mean(), 2),
        ))
    # chọn theo VAL: F1 trung bình, rồi AUC, rồi ít tham số
    rows.sort(key=lambda r: (-r["val_f1_mean"], -r["val_auc_mean"], r["params"]))
    return rows


def select_config(table, rule="1se"):
    """table đã sắp theo F1 val giảm dần -> (tên cấu hình chọn, các cấu hình coi là ngang nhau)."""
    best = table[0]
    if rule == "best":
        return best["config"], [best["config"]]
    floor = best["val_f1_mean"] - best["val_f1_std"]
    tied = [r for r in table if r["val_f1_mean"] >= floor]
    return min(tied, key=lambda r: (r["params"], -r["val_f1_mean"]))["config"], [r["config"] for r in tied]


def read_runs(path):
    ints = ("seed", "params", "best_epoch", "epochs_run")
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if k in ints:
                d[k] = int(v)
            else:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
        out.append(d)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tinh chỉnh B1 LSTM (chọn theo val)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--data", default="data/datasetv1.npz")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), help="tên cấu hình (mặc định: tất cả)")
    ap.add_argument("--seeds", type=int, default=3, help="số seed mỗi cấu hình (42, 43, ...)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default=None)
    ap.add_argument("--models-dir", default="models/b1")
    ap.add_argument("--out", default="output/b1_tuning")
    ap.add_argument("--select-to", default="models/b1_lstm.pt", help="chép checkpoint được chọn tới đây ('-' = không)")
    ap.add_argument("--rule", choices=("1se", "best"), default="1se", help="quy tắc chọn cấu hình (xem đầu file)")
    ap.add_argument("--reselect", action="store_true", help="không train, chọn lại từ <out>/runs.csv")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for k, (extra, desc) in CONFIGS.items():
            print(f"{k:9s} {desc}   {' '.join(extra)}")
        return 0
    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        ap.error(f"Không có cấu hình {unknown}. Có: {list(CONFIGS)}")

    models_dir, out_dir = ROOT / args.models_dir, ROOT / args.out
    models_dir.mkdir(parents=True, exist_ok=True)
    runs = read_runs(out_dir / "runs.csv") if args.reselect else []
    for name in ([] if args.reselect else args.configs):
        for seed in range(42, 42 + args.seeds):
            print(f"\n{'=' * 70}\n B1 | cấu hình {name} | seed {seed} | {CONFIGS[name][1]}\n{'=' * 70}")
            r = run_one(name, CONFIGS[name][0], seed, args.data, models_dir, args.device, args.epochs, args.patience)
            runs.append(r)
            write_csv(out_dir / "runs.csv", runs)              # ghi dần, ngắt giữa chừng vẫn còn kết quả

    table = summarize(runs)
    write_csv(out_dir / "configs.csv", table)
    best_cfg, tied = select_config(table, args.rule)
    best_run = max((r for r in runs if r["config"] == best_cfg), key=lambda r: (r["val_f1"], r["val_auc"]))

    print(f"\n{'=' * 70}\nTỔNG HỢP (sắp theo F1 val trung bình; test chỉ để tham khảo, KHÔNG dùng để chọn)\n")
    print(f"{'cấu hình':10s}{'tham số':>9s}{'F1 val':>15s}{'AUC val':>9s}{'F1 test':>15s}{'R test':>8s}{'P test':>8s}")
    for r in table:
        print(f"{r['config']:10s}{r['params']:>9,}{r['val_f1_mean']:>9.4f}±{r['val_f1_std']:.3f}"
              f"{r['val_auc_mean']:>9.4f}{r['test_f1_mean']:>9.4f}±{r['test_f1_std']:.3f}"
              f"{r['test_rec_mean']:>8.3f}{r['test_prec_mean']:>8.3f}")
    if args.rule == "1se":
        print(f"\nNgang nhau (trong 1 độ lệch chuẩn F1 val của {table[0]['config']}): {tied} -> chọn ít tham số nhất")
    print(f"Chọn: {best_cfg} seed {best_run['seed']} -> {best_run['checkpoint']}")

    if args.select_to and args.select_to != "-":
        src = ROOT / best_run["checkpoint"]
        dst = ROOT / args.select_to
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        shutil.copy2(src.with_suffix(".report.json"), dst.with_suffix(".report.json"))
        (out_dir / "selected.json").write_text(json.dumps(dict(best_run, selected_to=args.select_to,
                                                               rule=args.rule, tied_configs=tied,
                                                               description=CONFIGS[best_cfg][1]),
                                                          ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã chép thành {args.select_to}")
    print(f"Bảng: {out_dir / 'runs.csv'}, {out_dir / 'configs.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
