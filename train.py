#!/usr/bin/env python3
"""
train.py
--------
Huấn luyện bộ phân loại ngã (ST-GCN hoặc LSTM) trên dataset cửa sổ skeleton. Chạy được trên
máy cá nhân (VSCode, CPU hoặc GPU NVIDIA) lẫn Google Colab — không còn đường dẫn /content/drive cứng.

Bản này giữ nguyên cách huấn luyện của train.py (code_1):
  - trọng số lớp tính từ train, AdamW + CosineAnnealing, tăng cường lật ngang + nhiễu nhẹ,
  - chọn checkpoint theo (F1 lớp ngã, accuracy) trên VAL, early stopping,
  - chỉ đánh giá TEST sau khi đã chọn checkpoint.
Thêm:
  - --arch stgcn | lstm (mọi model đăng ký trong src/classifier.py),
  - đọc cả dataset code_1 (X_train/X_val/X_test) lẫn windows.npz của web,
  - chọn ngưỡng P(ngã) tốt nhất trên VAL (lưu vào checkpoint, web tự dùng),
  - lưu lịch sử từng epoch (.history.csv) và báo cáo (.report.json) cạnh checkpoint.

Ví dụ (chạy từ thư mục gốc project):
  python train.py --arch lstm
  python train.py --arch stgcn --out models/stgcn_local.pt
  python train.py --arch lstm --model-kw hidden=256 --model-kw bidirectional=true --epochs 100
  python train.py --arch lstm --data output/yolov8n-pose_10fps_filt/dataset/windows.npz

Gộp val vào train (không còn tập val; test giữ nguyên):
  python train.py --arch lstm --out models/best_lstm.pt                     # 1. chạy thường để biết epoch + ngưỡng
  python train.py --arch lstm --merge-val --from-report models/best_lstm.report.json --out models/lstm_trainval.pt
                                                                            # 2. train lại trên train+val
  (hoặc tự đặt: --merge-val --stop-epoch 25 --threshold 0.45)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_utils import (augment, best_threshold, check_split, describe, format_metrics,  # noqa: E402
                             load_splits, metrics_at, roc_auc, video_level)


def resolve_path(p) -> Path:
    """Đường dẫn tương đối: thử từ thư mục hiện tại, không có thì tính từ gốc project."""
    p = Path(p)
    return p if p.is_absolute() or p.exists() else ROOT / p


def parse_kw(items):
    """['hidden=256', 'bidirectional=true'] -> {'hidden': 256, 'bidirectional': True}."""
    import yaml
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--model-kw cần dạng key=value, nhận: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name=None):
    import torch
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_proba(model, X, device, batch_size=256):
    """X [N,T,17,3] -> P(ngã) [N]."""
    import torch
    from src.classifier import to_model_input
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            logits = model(to_model_input(X[s:s + batch_size]).to(device))
            out.append(torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def train(args) -> dict:
    import torch
    import torch.nn as nn
    from src.classifier import build_model, save_checkpoint, to_model_input

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    model_kw = parse_kw(args.model_kw)

    args.data = str(resolve_path(args.data))
    data = load_splits(args.data, fps=args.fps, seed=args.split_seed)
    stop_epoch, fixed_thr = args.stop_epoch, args.threshold
    if args.merge_val:
        # Gộp val vào train: không còn dữ liệu để chọn epoch / ngưỡng -> lấy từ lần chạy có val
        # (--from-report) hoặc đặt tay (--stop-epoch, --threshold). Test KHÔNG đổi.
        if args.from_report:
            rp = json.loads(resolve_path(args.from_report).read_text(encoding="utf-8"))
            stop_epoch = stop_epoch or int(rp["best_epoch"])
            fixed_thr = fixed_thr if fixed_thr is not None else float(rp.get("threshold", 0.5))
            if rp.get("arch") != args.arch:
                print(f"[cảnh báo] report là của {rp.get('arch')}, đang train {args.arch}")
            args.epochs = int((rp.get("args") or {}).get("epochs", args.epochs))   # giữ lịch lr giống lần đầu
        if not stop_epoch:
            raise SystemExit("--merge-val cần số epoch: --from-report <file .report.json> hoặc --stop-epoch N")
        for k in ("train", "val"):
            check_split(*data[k][:2], k)
        data["train"] = tuple(np.concatenate([data["train"][i], data["val"][i]]) for i in range(3))
        data["val"] = (data["val"][0][:0], data["val"][1][:0], data["val"][2][:0])
    (Xtr, ytr, vtr), (Xva, yva, vva), (Xte, yte, vte) = data["train"], data["val"], data["test"]
    has_val = len(yva) > 0
    for name, (X, y, _) in (("train", data["train"]), ("test", data["test"])) + ((("val", data["val"]),) if has_val else ()):
        check_split(X, y, name)
    if set(np.unique(ytr)) != {0, 1}:
        raise SystemExit("Train phải có cả lớp 0 và lớp 1.")
    for name, y in (("val", yva), ("test", yte)):
        if (name == "test" or has_val) and set(np.unique(y)) != {0, 1}:
            raise SystemExit(f"{name} cần có cả lớp 0 và lớp 1. Kiểm tra số subject và cách chia dataset.")
    last_epoch = min(args.epochs, stop_epoch) if stop_epoch else args.epochs

    out_path = Path(args.out or f"models/best_{args.arch}.pt")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset : {args.data} (định dạng {data['format']}, cửa sổ {data['window_size']} khung @ {data['fps']:g} fps)")
    if "auto_split" in data:
        print(f"Tự chia theo subject: {data['auto_split']}")
    print(describe(data) if has_val else "\n".join(l for l in describe(data).splitlines() if not l.startswith("val")))
    if args.merge_val:
        print(f"Gộp val vào train: train {last_epoch} epoch (lịch lr của {args.epochs} epoch), "
              f"ngưỡng {fixed_thr if fixed_thr is not None else 0.5:.2f}, không early stopping")
    print(f"Thiết bị: {device}")

    model = build_model(args.arch, in_channels=Xtr.shape[-1], num_classes=2, **model_kw).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model   : {args.arch} {model_kw or ''} | {n_params:,} tham số")

    n0, n1 = int((ytr == 0).sum()), int((ytr == 1).sum())
    weights = torch.tensor([len(ytr) / (2.0 * n0), len(ytr) / (2.0 * n1)], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def save(epoch, val_m, threshold):
        save_checkpoint(out_path, model, arch=args.arch, in_channels=int(Xtr.shape[-1]), num_classes=2,
                        model_kwargs=model_kw, window_size=int(data["window_size"]), fps=float(data["fps"]),
                        epoch=int(epoch), val_f1=None if val_m is None else float(val_m["f1"]),
                        val_acc=None if val_m is None else float(val_m["acc"]),
                        threshold=float(threshold), seed=int(args.seed), data=str(args.data),
                        merged_val=bool(args.merge_val))

    history = []
    best_key, best_epoch, bad = (-1.0, -1.0), 0, 0
    t_start = time.time()
    for epoch in range(1, last_epoch + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(Xtr))
        total = 0.0
        for s in range(0, len(order), args.batch_size):
            idx = order[s:s + args.batch_size]
            if len(idx) < 2:                    # BatchNorm cần >= 2 mẫu khi train
                continue
            xb = augment(Xtr[idx], rng, flip_p=0.0 if args.no_flip else 0.5, jitter_std=args.jitter)
            x = to_model_input(xb).to(device)
            yb = torch.from_numpy(ytr[idx]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), yb)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()

        if not has_val:                         # gộp val: chỉ theo dõi loss, lưu epoch cuối
            row = dict(epoch=epoch, loss=round(total / len(Xtr), 5), lr=optimizer.param_groups[0]["lr"],
                       seconds=round(time.time() - t0, 1))
            history.append(row)
            print(f"Epoch {epoch:03d}/{last_epoch} | loss {row['loss']:.4f} | {row['seconds']}s")
            continue
        p_val = predict_proba(model, Xva, device)
        val_m = metrics_at(yva, p_val, 0.5)
        row = dict(epoch=epoch, loss=round(total / len(Xtr), 5), lr=optimizer.param_groups[0]["lr"],
                   val_acc=round(val_m["acc"], 4), val_prec=round(val_m["prec"], 4),
                   val_rec=round(val_m["rec"], 4), val_f1=round(val_m["f1"], 4),
                   seconds=round(time.time() - t0, 1))
        history.append(row)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss {row['loss']:.4f} | val acc {val_m['acc']:.3f} | "
              f"P {val_m['prec']:.3f} | R {val_m['rec']:.3f} | F1 {val_m['f1']:.3f} | {row['seconds']}s")

        key = (val_m["f1"], val_m["acc"])
        if key > best_key:
            best_key, best_epoch, bad = key, epoch, 0
            save(epoch, val_m, 0.5)             # ngưỡng chọn lại sau khi có checkpoint tốt nhất
        else:
            bad += 1
        if bad >= args.patience:
            print(f"Early stopping tại epoch {epoch}.")
            break

    if not has_val:
        # ---- gộp val: dùng model ở epoch cuối, ngưỡng cố định
        best_epoch = last_epoch
        thr = float(fixed_thr) if fixed_thr is not None else 0.5
        save(best_epoch, None, thr)
        p_val = np.zeros(0)
    else:
        # ---- nạp checkpoint tốt nhất, chọn ngưỡng trên VAL, rồi mới đánh giá TEST
        from src.classifier import load_checkpoint
        ckpt = load_checkpoint(out_path, device)
        model.load_state_dict(ckpt["state"])
        p_val = predict_proba(model, Xva, device)
        thr = fixed_thr if fixed_thr is not None else (best_threshold(yva, p_val) if args.tune_threshold else 0.5)
        val_m = metrics_at(yva, p_val, 0.5)
        save(best_epoch, val_m, thr)

    p_test = predict_proba(model, Xte, device)
    rep = dict(
        arch=args.arch, model_kwargs=model_kw, params=n_params, checkpoint=str(out_path),
        data=str(args.data), data_format=data["format"], window_size=data["window_size"], fps=data["fps"],
        device=str(device), best_epoch=best_epoch, epochs_run=len(history),
        train_minutes=round((time.time() - t_start) / 60, 2), threshold=thr,
        merged_val=bool(args.merge_val),
        val_05=metrics_at(yva, p_val, 0.5) if has_val else None,
        val_thr=metrics_at(yva, p_val, thr) if has_val else None,
        val_auc=roc_auc(yva, p_val) if has_val else None,
        test_05=metrics_at(yte, p_test, 0.5), test_thr=metrics_at(yte, p_test, thr),
        test_auc=roc_auc(yte, p_test), test_video=video_level(yte, p_test, vte, thr),
        args={k: v for k, v in vars(args).items()},
    )
    stem = out_path.with_suffix("")
    with open(f"{stem}.history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    with open(f"{stem}.report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    if args.save_pred:
        np.savez_compressed(f"{stem}.test_pred.npz", p=p_test, y=yte, video=vte)

    print(f"\nCheckpoint      : {out_path}")
    if has_val:
        print(f"Epoch tốt nhất  : {best_epoch} | Val F1 (0.5) {val_m['f1']:.4f} | Val acc {val_m['acc']:.4f}")
        print(f"Ngưỡng chọn trên val: {thr:.2f}" + ("" if args.tune_threshold else " (không chọn, --no-tune-threshold)"))
    else:
        print(f"Đã train trên train+val {best_epoch} epoch, ngưỡng cố định {thr:.2f} (không có val để chọn)")
    print("\n===== KẾT QUẢ TEST (theo cửa sổ) =====")
    print(f"Ngưỡng 0.50 : {format_metrics(rep['test_05'])}")
    if thr != 0.5:
        print(f"Ngưỡng {thr:.2f} : {format_metrics(rep['test_thr'])}")
    if rep["test_auc"] is not None:
        print(f"AUC ROC     : {rep['test_auc']:.4f}")
    tv = rep["test_video"]
    print(f"Theo video ({tv['videos']} video, ngưỡng {thr:.2f}): {format_metrics(tv)}")
    print(f"\nĐã lưu: {stem}.history.csv, {stem}.report.json")
    return rep


def build_parser():
    ap = argparse.ArgumentParser(description="Huấn luyện ST-GCN / LSTM phát hiện ngã",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--arch", default="lstm", help="stgcn | lstm (mặc định lstm)")
    ap.add_argument("--data", default="data/dataset.npz", help="dataset code_1 hoặc windows.npz của web")
    ap.add_argument("--out", default=None, help="checkpoint ra (mặc định models/best_<arch>.pt)")
    ap.add_argument("--model-kw", action="append", default=[], metavar="KEY=VALUE",
                    help="tham số kiến trúc, ví dụ hidden=256, layers=1, bidirectional=true, velocity=false")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jitter", type=float, default=0.01, help="độ lệch chuẩn nhiễu tọa độ (0 = tắt)")
    ap.add_argument("--no-flip", action="store_true", help="tắt tăng cường lật ngang")
    ap.add_argument("--clip-grad", type=float, default=0.0, help="cắt gradient (gợi ý 1.0 cho LSTM; 0 = tắt)")
    ap.add_argument("--no-tune-threshold", dest="tune_threshold", action="store_false",
                    help="giữ ngưỡng 0.5 thay vì chọn trên val")
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:0 (mặc định tự chọn)")
    ap.add_argument("--fps", type=float, default=None, help="fps của dataset nếu file không lưu (code_1: 10)")
    ap.add_argument("--merge-val", action="store_true",
                    help="gộp tập val vào train (không còn val: không early stopping, không chọn ngưỡng)")
    ap.add_argument("--from-report", default=None,
                    help="với --merge-val: lấy số epoch tốt nhất + ngưỡng từ .report.json của lần chạy có val")
    ap.add_argument("--stop-epoch", type=int, default=None,
                    help="dừng ở epoch này (lịch lr vẫn theo --epochs); dùng với --merge-val")
    ap.add_argument("--threshold", type=float, default=None, help="ngưỡng P(ngã) cố định thay vì chọn trên val")
    ap.add_argument("--split-seed", type=int, default=0, help="seed chia subject khi windows.npz chưa có split")
    ap.add_argument("--save-pred", action="store_true", help="lưu P(ngã) của test để phân tích")
    return ap


if __name__ == "__main__":
    _args = build_parser().parse_args()
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("Chưa cài PyTorch: pip install torch  (xem HUONG_DAN_B1.md mục 2)")
    train(_args)
