#!/usr/bin/env python3
"""
app_check.py
------------
Kiểm tra kết nối model với App: chạy video THẬT qua đúng pipeline của web/main.py
    OpenedSource (10 fps) -> YOLO pose -> SkeletonFilter -> FallClassifier.update() / flush()
cho nhiều model cùng lúc (cùng một chuỗi pose), rồi so với bộ test chung:

  1. NHẬN DỮ LIỆU: cửa sổ skeleton App gom được có giống cửa sổ trong bộ test của cùng video không
     (sai khác trung bình trên tọa độ đã chuẩn hóa; lớn = cấu hình pose của App khác lúc trích dataset).
  2. KẾT QUẢ: dự đoán video của App (có >= 1 cửa sổ báo NGÃ) so với nhãn thật và với dự đoán offline
     trên testset (test_tool.py) — hai bên phải khớp nhau.
  3. TỐC ĐỘ: FPS cả pipeline, thời gian pose và thời gian phân loại mỗi lần dự đoán.

Chạy (mặc định: configs/v1.yaml, B1 = models/b1_lstm.pt, B2 = models/shiftgcn_best.pt, 12 video của
subject test lấy đều từ data/videos_new):
    python app_check.py
    python app_check.py --limit 0                       # mọi video của subject test (chậm)
    python app_check.py --videos data/videos/fall/sub3_02.mp4 data/videos/adl/sub1_01.mp4
    python app_check.py --save-dir output/app_check/videos   # lưu video có vẽ skeleton + nhãn để xem
Ra: output/app_check/<thời gian>/{app_predictions.csv, summary.json, report.md}
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
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

from web.config import load_config, resolve  # noqa: E402

FILE_RE = re.compile(r"^(SBJ\d+)_(LOC\d+)_(ACT\d+)_([A-Z])_(\d+)$", re.IGNORECASE)


def dataset_video_id(path: Path) -> str | None:
    """data/videos_new/fall/SBJ13_LOC2_ACT10_B_1.mp4 -> 'SBJ13_Fall_LOC2-ACT10-B-1' (tên trong datasetv1)."""
    m = FILE_RE.match(path.stem)
    if not m:
        return None
    kind = "Fall" if path.parent.name.lower() == "fall" else "ADL"
    return f"{m[1].upper()}_{kind}_{m[2].upper()}-{m[3].upper()}-{m[4].upper()}-{m[5]}"


def pick_videos(root: Path, subjects, limit: int) -> list[Path]:
    vids = sorted(p for kind in ("fall", "adl") for p in (root / kind).glob("*.mp4")
                  if p.stem.split("_")[0].upper() in subjects)
    if limit and len(vids) > limit:                       # lấy đều, nửa ngã nửa không ngã
        fall = [v for v in vids if v.parent.name == "fall"]
        adl = [v for v in vids if v.parent.name != "fall"]
        k = limit // 2
        take = lambda xs, n: [xs[i] for i in np.linspace(0, len(xs) - 1, n).round().astype(int)] if xs else []  # noqa: E731
        vids = take(fall, k) + take(adl, limit - k)
    return vids


def run_video(path, cfg, clfs: dict, save_dir=None):
    from src.skeleton_io import from_poses
    from web.engine import OpenedSource, build_drawer, build_estimator, build_filter, draw_fall, estimate

    est, flt = run_video.cache.get("est"), run_video.cache.get("flt")
    if est is None:
        est, flt = build_estimator(cfg), build_filter(cfg)
        run_video.cache.update(est=est, flt=flt, drawer=build_drawer(cfg))
    drawer = run_video.cache["drawer"]
    est.reset()
    if flt is not None:
        flt.reset()
    for c in clfs.values():
        c.reset()
    out = {k: [] for k in clfs}
    poses, pose_ms, clf_ms = [], [], {k: [] for k in clfs}
    writer = None
    t0 = time.perf_counter()
    with OpenedSource(str(path), cfg) as src:
        for pkt in src.frames():
            pose, lat = estimate(est, flt, pkt.frame, pkt.idx, pkt.t)
            pose_ms.append(lat)
            poses.append(pose)
            res_main = None
            for k, c in clfs.items():
                t1 = time.perf_counter()
                r = c.update(pose)
                if r is not None:
                    clf_ms[k].append((time.perf_counter() - t1) * 1000)
                    out[k].append(r)
                    res_main = res_main or r
            if save_dir is not None:
                import cv2
                img = drawer.draw(pkt.frame, pose)
                draw_fall(img, res_main or next((c.last for c in clfs.values() if c.last), None))
                if writer is None:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(str(Path(save_dir) / f"{Path(path).stem}.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), src.out_fps or 10, (w, h))
                writer.write(img)
    for k, c in clfs.items():                              # video ngắn hơn cửa sổ: dự đoán một lần (đệm cuối)
        r = c.flush()
        if r is not None:
            out[k].append(r)
    if writer is not None:
        writer.release()
    wall = time.perf_counter() - t0
    item = from_poses(poses) if poses else None
    return dict(results=out, frames=len(poses), wall_s=wall, pose_ms=float(np.mean(pose_ms)) if pose_ms else None,
                clf_ms={k: (float(np.mean(v)) if v else None) for k, v in clf_ms.items()}, item=item)


run_video.cache = {}


def app_windows(item, W):
    """Cửa sổ chuẩn hóa App đưa vào model (stride 1), giống FallClassifier.update + flush."""
    from src.classifier import pad_to_window
    from src.features import normalize_clip
    seq, mask = item["seq"], item["mask"]
    if len(seq) < W:
        s, m = pad_to_window(seq, mask, W)
        return np.stack([normalize_clip(s, m, item["aspect"])])
    return np.stack([normalize_clip(seq[e - W + 1:e + 1], mask[e - W + 1:e + 1], item["aspect"])
                     for e in range(W - 1, len(seq))])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kiểm tra model trong App (pipeline web) so với bộ test chung",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--model", action="append", default=[], metavar="TÊN=ĐƯỜNG_DẪN",
                    help="mặc định B1=models/b1_lstm.pt, B2=models/shiftgcn_best.pt")
    ap.add_argument("--videos", nargs="*", default=None, help="video cụ thể (mặc định: subject test)")
    ap.add_argument("--video-root", default="data/videos_new")
    ap.add_argument("--testset", default="data/testsets/v1_test")
    ap.add_argument("--limit", type=int, default=12, help="số video lấy từ subject test (0 = tất cả)")
    ap.add_argument("--save-dir", default=None, help="lưu video có vẽ kết quả")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from src.classifier import FallClassifier
    from src.testset import load_testset

    cfg = load_config(resolve(args.config), args.set)
    specs = dict(m.split("=", 1) for m in args.model) or {"B1": "models/b1_lstm.pt", "B2": "models/shiftgcn_best.pt"}
    c = cfg["classifier"]
    clfs = {k: FallClassifier(resolve(p), stride_frames=c.get("stride_frames", 1), device=c.get("device"),
                              fall_threshold=c.get("threshold"), speed_gate=c.get("speed_gate"), verbose=False)
            for k, p in specs.items()}
    W = {k: v.window_size for k, v in clfs.items()}

    ts = load_testset(resolve(args.testset))
    subjects = set(ts["manifest"].get("subjects") or set(ts["subject"]))
    videos = [Path(v) for v in args.videos] if args.videos else pick_videos(resolve(args.video_root), subjects, args.limit)
    if not videos:
        sys.exit("Không có video để kiểm tra.")
    # dự đoán offline trên testset (giống test_tool.py) để so khớp
    offline = {}
    for k, clf in clfs.items():
        p = clf.predict_batch(ts["X"]) if W[k] == ts["X"].shape[1] else None
        offline[k] = p

    print(f"Cấu hình {args.config} | pose {cfg['pose']['model_path']} imgsz {cfg['pose'].get('imgsz')} | "
          f"model {', '.join(f'{k}={clfs[k].name} (ngưỡng {clfs[k].fall_threshold:.2f})' for k in clfs)}")
    rows = []
    for i, v in enumerate(videos, 1):
        vid = dataset_video_id(v)
        label = 1 if v.parent.name.lower() == "fall" else 0
        r = run_video(v, cfg, clfs, args.save_dir and resolve(args.save_dir))
        row = dict(video=str(v).replace("\\", "/"), dataset_id=vid or "", label=label, frames=r["frames"],
                   fps=round(r["frames"] / r["wall_s"], 1) if r["wall_s"] else None,
                   pose_ms=round(r["pose_ms"], 1) if r["pose_ms"] else None)
        m = ts["video"] == vid if vid else np.zeros(len(ts["y"]), bool)
        in_test = bool(m.any())
        row["in_testset"] = int(in_test)
        if in_test and r["item"] is not None:
            aw = app_windows(r["item"], ts["X"].shape[1])
            tw = ts["X"][m]
            n = min(len(aw), len(tw))
            row["skeleton_diff"] = round(float(np.abs(aw[:n, ..., :2] - tw[:n, ..., :2]).mean()), 4)
            row["mask_agree"] = round(float((aw[:n, ..., 2] == tw[:n, ..., 2]).mean()), 3)
        for k in clfs:
            res = r["results"][k]
            probs = [x["probability"] for x in res]
            row[f"{k}_n_pred"] = len(res)
            row[f"{k}_max_p"] = round(max(probs), 4) if probs else None
            row[f"{k}_app_pred"] = int(any(x["label"] for x in res))
            row[f"{k}_clf_ms"] = round(r["clf_ms"][k], 2) if r["clf_ms"][k] else None
            if in_test and offline[k] is not None:
                row[f"{k}_test_max_p"] = round(float(offline[k][m].max()), 4)
                row[f"{k}_test_pred"] = int((offline[k][m] >= clfs[k].fall_threshold).any())
        rows.append(row)
        print(f"[{i}/{len(videos)}] {v.name:32s} nhãn={'NGÃ' if label else 'ADL':3s} khung={r['frames']:3d} "
              f"lệch skeleton={row.get('skeleton_diff', '-')} | " +
              " | ".join(f"{k}: App {'NGÃ' if row[f'{k}_app_pred'] else 'ADL'} p={row[f'{k}_max_p']}"
                         f" (test {row.get(f'{k}_test_max_p', '-')})" for k in clfs))

    # ---- tổng hợp
    from src.train_utils import metrics
    summary = dict(created_at=datetime.now().astimezone().isoformat(timespec="seconds"), config=args.config,
                   pose=cfg["pose"]["model_path"], imgsz=cfg["pose"].get("imgsz"), videos=len(rows),
                   fps_mean=round(float(np.mean([r["fps"] for r in rows if r["fps"]])), 1),
                   pose_ms_mean=round(float(np.mean([r["pose_ms"] for r in rows if r["pose_ms"]])), 1), models={})
    in_test = [r for r in rows if r["in_testset"]]
    if in_test:
        summary["skeleton_diff_mean"] = round(float(np.mean([r["skeleton_diff"] for r in in_test])), 4)
        summary["mask_agree_mean"] = round(float(np.mean([r["mask_agree"] for r in in_test])), 3)
    for k in clfs:
        m = metrics([r["label"] for r in rows], [r[f"{k}_app_pred"] for r in rows])
        d = dict(checkpoint=specs[k], arch=clfs[k].arch, threshold=clfs[k].fall_threshold,
                 app_video=dict(prec=round(m["prec"], 4), rec=round(m["rec"], 4), f1=round(m["f1"], 4),
                                tp=m["tp"], fp=m["fp"], fn=m["fn"], tn=m["tn"]),
                 clf_ms_mean=round(float(np.mean([r[f"{k}_clf_ms"] for r in rows if r[f"{k}_clf_ms"]])), 2))
        both = [r for r in in_test if f"{k}_test_pred" in r]
        if both:
            d["agree_with_testset"] = f"{sum(r[f'{k}_app_pred'] == r[f'{k}_test_pred'] for r in both)}/{len(both)}"
            d["prob_diff_mean"] = round(float(np.mean([abs(r[f"{k}_max_p"] - r[f"{k}_test_max_p"]) for r in both])), 4)
        summary["models"][k] = d

    out = resolve(args.out or f"output/app_check/{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({f for r in rows for f in r}, key=lambda f: list(rows[0]).index(f) if f in rows[0] else 99)
    with open(out / "app_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    L = [f"# Kiểm tra model trong App — {summary['created_at']}", "",
         f"- Cấu hình `{args.config}`: pose `{summary['pose']}` imgsz {summary['imgsz']}, {len(rows)} video",
         f"- Tốc độ cả pipeline: **{summary['fps_mean']} khung/s** (pose {summary['pose_ms_mean']} ms/khung)"]
    if in_test:
        L.append(f"- Nhận dữ liệu: lệch tọa độ chuẩn hóa so với bộ test **{summary['skeleton_diff_mean']}** "
                 f"(đơn vị độ dài xương), mask khớp {summary['mask_agree_mean']:.1%} ({len(in_test)} video có trong bộ test)")
    L += ["", "| Model | Kiến trúc | Ngưỡng | P (video) | R (video) | F1 (video) | FN | FP | Khớp testset | Lệch P(ngã) | ms/lần dự đoán |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, d in summary["models"].items():
        a = d["app_video"]
        L.append(f"| {k} | {d['arch']} | {d['threshold']:.2f} | {a['prec']} | {a['rec']} | {a['f1']} | {a['fn']} | "
                 f"{a['fp']} | {d.get('agree_with_testset', '-')} | {d.get('prob_diff_mean', '-')} | {d['clf_ms_mean']} |")
    L += ["", "| Video | Nhãn | Lệch skeleton | " + " | ".join(f"{k} App (p max) | {k} testset" for k in clfs) + " |",
          "|---|---|---|" + "---|---|" * len(clfs)]
    for r in rows:
        L.append(f"| `{Path(r['video']).name}` | {'NGÃ' if r['label'] else 'ADL'} | {r.get('skeleton_diff', '-')} | " +
                 " | ".join(f"{'NGÃ' if r[f'{k}_app_pred'] else 'ADL'} ({r[f'{k}_max_p']}) | "
                            f"{r.get(f'{k}_test_max_p', '-')}" for k in clfs) + " |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L[:5]))
    for k, d in summary["models"].items():
        print(f"  {k}: video F1 {d['app_video']['f1']} | khớp testset {d.get('agree_with_testset', '-')} | "
              f"lệch P {d.get('prob_diff_mean', '-')} | {d['clf_ms_mean']} ms/lần")
    print(f"Kết quả: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
