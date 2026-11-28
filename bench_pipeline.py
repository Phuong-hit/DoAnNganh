#!/usr/bin/env python3
"""Đo hiệu năng MỘT pipeline hoàn chỉnh (đề cương mục 3.3): End-to-end latency, End-to-end FPS, Peak RAM.

Pipeline: lấy khung -> pose (YOLO) -> SkeletonFilter -> cửa sổ 17 khung -> model phân loại -> hậu xử lý -> NGÃ/KHÔNG.
Dùng đúng cấu hình web (configs/web.yaml): model, ngưỡng, hậu xử lý của model đang chọn.

  python bench_pipeline.py --source 0 --label pc_gpu                       # webcam PC
  python bench_pipeline.py --source 0 --label pc_cpu --cpu                 # ép chạy CPU
  python bench_pipeline.py --source data/videos/fall/sub3_02.mp4 --label pc_video
  python bench_pipeline.py --source maix://10.123.219.1 --label maix_split # pose trên MaixCAM, phân loại trên PC
  python bench_pipeline.py --source 0 --checkpoint models/b1_lstm.pt --label b1_lstm

Kết quả: output/bench/<label>_<thời gian>/frames.csv (từng khung) + summary.json,
và một dòng thêm vào output/bench/summary.csv (gộp mọi lần đo, mở bằng Excel để làm bảng báo cáo).

Các cột thời gian (ms, mỗi khung):
  read_ms    chờ/đọc khung (webcam: chờ camera; file: giải mã; maix: chờ khung qua mạng + giải mã JPEG)
  pose_ms    YOLO-pose (maix: thời gian đo TRÊN MaixCAM)
  filter_ms  SkeletonFilter
  model_ms   forward của model phân loại (0 ở khung không dự đoán)
  post_ms    gom cửa sổ + chuẩn hóa + hậu xử lý (phần còn lại của bước phân loại)
  e2e_ms     từ lúc có khung tới lúc ra quyết định = pose + filter + model + post
  loop_ms    cả vòng, tính cả read_ms  -> e2e_fps = 1000 / mean(loop_ms)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.config import load_config, resolve  # noqa: E402

COLS = ["read_ms", "pose_ms", "filter_ms", "model_ms", "post_ms", "e2e_ms", "loop_ms"]


class Memory:
    """RAM của tiến trình: RSS hiện tại, đỉnh (peak) do hệ điều hành ghi nhận, và đỉnh lấy mẫu mỗi khung."""

    def __init__(self):
        import psutil
        self.proc = psutil.Process()
        self.sampled_peak = 0

    def rss_mb(self) -> float:
        rss = self.proc.memory_info().rss
        self.sampled_peak = max(self.sampled_peak, rss)
        return rss / 2**20

    def peak_mb(self) -> float:
        info = self.proc.memory_info()
        if hasattr(info, "peak_wset"):                  # Windows: đỉnh working set của cả tiến trình
            return info.peak_wset / 2**20
        try:
            import resource                             # Linux: ru_maxrss tính bằng KB
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            return self.sampled_peak / 2**20


def stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {}
    return dict(mean=round(float(x.mean()), 2), p50=round(float(np.percentile(x, 50)), 2),
                p95=round(float(np.percentile(x, 95)), 2), max=round(float(x.max()), 2))


def frames_cv2(spec: str, target_fps: float | None):
    """Webcam / file bằng OpenCV, KHÔNG tự chờ cho đúng nhịp (đo năng lực xử lý tối đa).
    File: lấy mẫu về target_fps bằng cách bỏ khung (grab) như main.py nhánh final."""
    import cv2
    is_cam = spec.isdigit()
    cap = cv2.VideoCapture(int(spec) if is_cam else str(resolve(spec) if not Path(spec).exists() else spec))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn {spec}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1 if is_cam or not target_fps else max(1, round(src_fps / target_fps))
    k = 0
    try:
        while True:
            t0 = time.perf_counter()
            for _ in range(step - 1):
                cap.grab()
            ok, frame = cap.read()
            if not ok:
                return
            yield k, k / (target_fps or src_fps), frame, None, 0.0, (time.perf_counter() - t0) * 1000
            k += 1
    finally:
        cap.release()


def frames_maix(spec: str):
    from web.maixcam import MaixPoseSource
    src = MaixPoseSource(spec, target_fps=1000.0)      # không giới hạn: lấy mọi khung MaixCAM gửi
    it = iter(src.get_frames())
    try:
        while True:
            t0 = time.perf_counter()
            try:
                pkt = next(it)
            except StopIteration:
                if src.error:
                    print("MaixCAM:", src.error)
                return
            yield pkt.idx, pkt.t, pkt.frame, pkt.pose, pkt.pose_ms, (time.perf_counter() - t0) * 1000
    finally:
        src.release()


def maix_info(spec: str) -> dict | None:
    """Số liệu MaixCAM tự đo (maixcam_app/main.py: /bench) — thời gian từng bước và RAM trên board."""
    import urllib.request
    from urllib.parse import urlparse
    u = urlparse(spec)
    for path in ("/bench", "/info"):
        try:
            with urllib.request.urlopen(f"http://{u.hostname}:{u.port or 8000}{path}", timeout=3) as r:
                return json.loads(r.read().decode())
        except Exception:
            continue
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="0 = webcam | đường dẫn video | maix://<IP>")
    ap.add_argument("--frames", type=int, default=1000, help="số khung ĐO (sau warm-up), mặc định 1000")
    ap.add_argument("--warmup", type=int, default=30, help="số khung đầu bỏ qua (nạp model, cache...)")
    ap.add_argument("--label", default=None, help="tên lần đo, ví dụ pc_gpu / pc_cpu / maix_split")
    ap.add_argument("--checkpoint", default=None, help="model phân loại (mặc định: classifier.checkpoint)")
    ap.add_argument("--cpu", action="store_true", help="ép pose + phân loại chạy CPU")
    ap.add_argument("--no-filter", action="store_true", help="tắt SkeletonFilter")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="ghi đè cấu hình như run_web.py")
    args = ap.parse_args(argv)

    mem = Memory()
    ram_start = mem.rss_mb()

    cfg = load_config(args.config, args.set)
    if args.checkpoint:
        cfg["classifier"]["checkpoint"] = args.checkpoint
    if args.cpu:
        cfg["pose"]["device"] = "cpu"
        cfg["classifier"]["device"] = "cpu"
    if args.no_filter:
        cfg.setdefault("filter", {})["enabled"] = False

    from web.engine import build_classifier, build_estimator, build_filter
    is_maix = args.source.lower().startswith("maix://")
    estimator = None if is_maix else build_estimator(cfg)
    flt = build_filter(cfg)
    clf = build_classifier(cfg)
    if clf is None:
        raise SystemExit("Chưa có model phân loại (classifier.checkpoint).")
    # warm-up model phân loại trước khi đo (lần gọi đầu trên GPU mất hàng trăm ms do khởi tạo CUDA)
    dummy = np.zeros((1, clf.window_size, 17, 3), np.float32)
    for _ in range(5):
        clf.predict_batch(dummy)
    ram_loaded = mem.rss_mb()

    # đo riêng thời gian forward của model: bọc predict_batch
    model_t = [0.0]
    orig = clf.predict_batch

    def timed_predict(clips):
        t = time.perf_counter()
        out = orig(clips)
        model_t[0] += (time.perf_counter() - t) * 1000
        return out
    clf.predict_batch = timed_predict

    cuda = False
    try:
        import torch
        cuda = clf.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    label = args.label or ("maix" if is_maix else "pc") + ("_cpu" if args.cpu else "")
    out_dir = resolve("output/bench") / f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
    source = frames_maix(args.source) if is_maix else frames_cv2(args.source, cfg["source"].get("target_fps"))

    rows, n_total, n_fall, n_pred = [], 0, 0, 0
    total = args.warmup + args.frames
    print(f"Đo {label}: {args.frames} khung (+{args.warmup} warm-up) | model {clf.name} | {clf.device}")
    wall0 = None
    t_prev_end = time.perf_counter()
    for idx, t, frame, pose, board_pose_ms, read_ms in source:
        t_a = time.perf_counter()
        if pose is None:
            pose = estimator.process_frame(frame, idx, t)
            pose_ms = (time.perf_counter() - t_a) * 1000
        else:
            pose_ms = board_pose_ms                    # pose đã chạy trên MaixCAM
        t_b = time.perf_counter()
        if flt is not None:
            pose = flt.smooth(pose)
        t_c = time.perf_counter()
        model_t[0] = 0.0
        res = clf.update(pose)
        t_d = time.perf_counter()
        filter_ms = (t_c - t_b) * 1000
        model_ms = model_t[0]
        post_ms = (t_d - t_c) * 1000 - model_ms
        pc_pose_ms = (t_b - t_a) * 1000
        e2e = (pose_ms if is_maix else pc_pose_ms) + filter_ms + model_ms + post_ms
        loop_ms = (t_d - t_prev_end) * 1000
        t_prev_end = t_d
        ram = mem.rss_mb()
        n_total += 1
        if n_total <= args.warmup:
            continue
        if wall0 is None:
            wall0 = t_d - loop_ms / 1000
            mem.sampled_peak = 0                       # đỉnh RAM chỉ tính trong lúc đo
        n_pred += int(res is not None and model_ms > 0)
        n_fall += int(bool(res and res["label"]))
        rows.append([round(read_ms, 3), round(pose_ms, 3), round(filter_ms, 3), round(model_ms, 3),
                     round(post_ms, 3), round(e2e, 3), round(loop_ms, 3), round(ram, 1),
                     int(bool(pose.valid)), None if res is None else round(res["probability"], 4),
                     None if res is None else int(res["label"])])
        if len(rows) % 100 == 0:
            print(f"  {len(rows)}/{args.frames} khung | e2e {np.mean([r[5] for r in rows]):.1f} ms "
                  f"| RAM {ram:.0f} MB")
        if n_total >= total:
            break
    if estimator is not None:
        estimator.close()
    if not rows:
        raise SystemExit("Không đo được khung nào (nguồn rỗng hoặc ngắn hơn warm-up).")
    wall = time.perf_counter() - wall0

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS + ["ram_mb", "person", "p_fall", "label"])
        w.writerows(rows)

    a = np.array([r[:7] for r in rows], dtype=float)
    per = {c: stats(a[:, i]) for i, c in enumerate(COLS)}
    model_rows = a[a[:, 3] > 0, 3]
    per["model_ms"] = stats(model_rows)               # chỉ tính các khung có chạy model
    summary = dict(
        label=label, time=datetime.now().isoformat(timespec="seconds"), source=args.source,
        frames=len(rows), warmup=args.warmup, wall_s=round(wall, 2),
        e2e_fps=round(len(rows) / wall, 2),
        compute_fps=round(1000 / per["e2e_ms"]["mean"], 2) if per["e2e_ms"]["mean"] else None,
        latency_ms=per,
        ram_mb=dict(start=round(ram_start, 1), after_load=round(ram_loaded, 1),
                    peak=round(mem.sampled_peak / 2**20, 1),          # đỉnh RSS trong lúc pipeline chạy
                    process_peak=round(mem.peak_mb(), 1),             # đỉnh cả đời tiến trình (gồm lúc nạp thư viện)
                    pipeline=round(mem.sampled_peak / 2**20 - ram_start, 1)),
        gpu_peak_mb=None,
        person_ratio=round(float(np.mean([r[8] for r in rows])), 3), predictions=n_pred, fall_frames=n_fall,
        config=dict(pose_model=None if is_maix else str(cfg["pose"].get("model_path")),
                    imgsz=cfg["pose"].get("imgsz"), classifier=clf.name, params=clf.num_params,
                    window=clf.window_size, stride=clf.stride_frames, threshold=clf.fall_threshold,
                    postprocess=clf.post is not None, filter=flt is not None, device=str(clf.device),
                    target_fps=cfg["source"].get("target_fps")),
        machine=dict(os=platform.platform(), cpu=platform.processor(), python=platform.python_version()),
    )
    if cuda:
        import torch
        summary["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        summary["machine"]["gpu"] = torch.cuda.get_device_name(0)
    if is_maix:
        summary["maixcam"] = maix_info(args.source)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # một dòng / lần đo trong bảng gộp
    agg = resolve("output/bench/summary.csv")
    flat = dict(label=label, time=summary["time"], source=args.source, frames=len(rows),
                classifier=clf.name, device=str(clf.device), e2e_fps=summary["e2e_fps"],
                compute_fps=summary["compute_fps"],
                **{f"{c}_{k}": per[c].get(k) for c in COLS for k in ("mean", "p95")},
                ram_peak_mb=summary["ram_mb"]["peak"], ram_process_peak_mb=summary["ram_mb"]["process_peak"],
                ram_pipeline_mb=summary["ram_mb"]["pipeline"],
                gpu_peak_mb=summary["gpu_peak_mb"])
    mc = summary.get("maixcam") or {}
    for k in ("pose_ms_mean", "pose_ms_p95", "loop_ms_mean", "fps", "vm_hwm_mb", "mem_used_peak_mb",
              "mem_used_before_mb", "mem_total_mb"):
        flat[f"maix_{k}"] = mc.get(k)
    new = not agg.exists()
    with open(agg, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(flat))
        if new:
            w.writeheader()
        w.writerow(flat)

    print(f"\n=== {label}: {len(rows)} khung, {wall:.1f} s ===")
    print(f"{'bước':10s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'max':>8s}   (ms)")
    for c in COLS:
        s = per[c]
        if s:
            print(f"{c:10s} {s['mean']:8.2f} {s['p50']:8.2f} {s['p95']:8.2f} {s['max']:8.2f}")
    print(f"End-to-end FPS: {summary['e2e_fps']}  (chỉ tính xử lý: {summary['compute_fps']})")
    r = summary["ram_mb"]
    print(f"RAM: lúc đầu {r['start']} MB | sau nạp model {r['after_load']} MB | PEAK khi chạy {r['peak']} MB "
          f"| pipeline dùng thêm {r['pipeline']} MB | đỉnh cả tiến trình {r['process_peak']} MB" + (f" | GPU peak {summary['gpu_peak_mb']} MB" if cuda else ""))
    if is_maix:
        print("MaixCAM tự đo:", json.dumps(summary.get("maixcam"), ensure_ascii=False))
    print(f"Đã lưu: {out_dir}  và  {agg}")


if __name__ == "__main__":
    main()
