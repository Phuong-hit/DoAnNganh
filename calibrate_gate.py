#!/usr/bin/env python3
"""
calibrate_gate.py
-----------------
Chọn ngưỡng CỔNG TỐC ĐỘ (src/motion.py) cho một checkpoint, để nằm xuống / leo lên giường / ngả lưng chậm
không bị báo ngã, mà KHÔNG làm mất cú ngã nào.

  1. Chạy model trên tập VAL (mặc định), tính P(ngã) và tốc độ (vai hạ, thân xoay) mỗi cửa sổ.
  2. Thử mọi tổ hợp ngưỡng; chỉ giữ tổ hợp không làm giảm số video ngã phát hiện được
     (--max-recall-drop 0), chọn tổ hợp F1 cao nhất / ít video sinh hoạt bị báo nhầm nhất.
  3. Báo cáo trên TEST (chỉ để xem, không dùng để chọn).
  4. --save: ghi ngưỡng vào checkpoint ("speed_gate"); web / main.py tự dùng khi classifier.speed_gate = null.

  python calibrate_gate.py models/best_lstm.pt
  python calibrate_gate.py models/best_lstm.pt --data data/dataset_clean.npz --save
  python calibrate_gate.py models/lstm_trainval.pt --split train      # model đã gộp val: không còn val

Lưu ý: tập val chỉ có một người (Subject 1). Ngưỡng nên kiểm tra lại trên video thật của nhóm
(web: ô "Cổng tốc độ" chỉnh được trực tiếp).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.motion import SpeedGate, motion_features, search_gate  # noqa: E402
from src.train_utils import format_metrics, load_splits, metrics, video_level  # noqa: E402


def resolve_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() or p.exists() else ROOT / p


def fmt_gate(g: dict) -> str:
    rot = "không dùng" if g["rotation"] >= 1e8 else f"{g['rotation']:g}°/s"
    keep = "không" if g["keep_if_p_above"] is None else f"{g['keep_if_p_above']:g}"
    return f"vai ≥ {g['shoulder_drop']:g} thân/s | xoay ≥ {rot} | luôn báo nếu P ≥ {keep}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chọn ngưỡng cổng tốc độ trên tập val",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--split", default="val", choices=["train", "val"], help="tập dùng để CHỌN ngưỡng")
    ap.add_argument("--max-recall-drop", type=float, default=0.0,
                    help="cho phép recall theo video giảm tối đa bao nhiêu (0 = không được mất video ngã nào)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", action="store_true", help="ghi ngưỡng tốt nhất vào checkpoint")
    args = ap.parse_args(argv)
    try:
        import torch
    except ImportError:
        sys.exit("Chưa cài PyTorch: pip install torch")
    from src.classifier import FallClassifier, load_checkpoint

    ck = resolve_path(args.checkpoint)
    clf = FallClassifier(ck, device=args.device, verbose=False, speed_gate={"enabled": False})
    data = load_splits(resolve_path(args.data))
    X, y, vid = data[args.split]
    p = clf.predict_batch(X)
    feats = motion_features(X, data["fps"])
    res = search_gate(y, p, clf.fall_threshold, feats, vid, args.max_recall_drop)
    base, cands = res[0]["baseline"], res[1:]
    print(f"Model {clf.name}, ngưỡng P {clf.fall_threshold:.2f}. Chọn trên tập {args.split}: {len(y)} cửa sổ, "
          f"{len(set(vid))} video.\n")
    print(f"Không có cổng : F1 {base['f1']:.3f} | R {base['rec']:.3f} | P {base['prec']:.3f} | "
          f"video ngã phát hiện {base['video_rec']:.0%} | video sinh hoạt báo nhầm {base['video_fp']}")
    if not cands:
        print("Không có tổ hợp nào giữ được recall theo video. Thử --max-recall-drop 0.05.")
        return 1
    print(f"\nTốt nhất trên {args.split} (không làm mất video ngã quá {args.max_recall_drop:.0%}):")
    for r in cands[:args.top]:
        print(f"  F1 {r['f1']:.3f} | R {r['rec']:.3f} | P {r['prec']:.3f} | video báo nhầm {r['video_fp']:2d} | "
              f"video ngã {r['video_rec']:.0%} | {fmt_gate(r['gate'])}")
    best = cands[0]["gate"]
    gate = SpeedGate.from_dict(best)

    Xt, yt, vt = data["test"]
    pt = clf.predict_batch(Xt)
    ft = motion_features(Xt, data["fps"])
    pred0 = (pt >= clf.fall_threshold).astype(int)
    pred1 = (pred0.astype(bool) & gate.passes(ft, pt)).astype(int)
    print("\nTEST (chỉ để xem, KHÔNG dùng để chọn):")
    for label, pr in (("không cổng", pred0), ("có cổng  ", pred1)):
        mv = video_level(yt, pr.astype(float), vt, 0.5)
        print(f"  {label}: {format_metrics(metrics(yt, pr))} | video ngã {mv['rec']:.0%} | video báo nhầm {mv['fp']}")
    print(f"\nĐề xuất: {fmt_gate(best)}")
    if args.save:
        ckpt = load_checkpoint(ck, "cpu")
        ckpt["speed_gate"] = dict(gate.to_dict(), chosen_on=args.split)
        torch.save(ckpt, ck)
        print(f"Đã ghi vào {ck} (khóa speed_gate). Web / main.py dùng khi classifier.speed_gate = null.")
    else:
        print("Thêm --save để ghi vào checkpoint, hoặc đặt trong configs/*.yaml:\n"
              f"  speed_gate: {{enabled: true, shoulder_drop: {gate.shoulder_drop:g}, rotation: {gate.rotation:g}, "
              f"keep_if_p_above: {gate.keep_if_p_above if gate.keep_if_p_above is not None else 'null'}}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
