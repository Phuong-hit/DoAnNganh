#!/usr/bin/env python3
"""Bảng hiệu năng theo TỪNG THÀNH PHẦN cho báo cáo (kiểu bảng Metric | Value):

  Pose (YOLO-Pose)     Parameters, Model size, Latency (ms/frame), Throughput (FPS), Peak VRAM / Peak RAM
  Bộ phân loại ngã     Parameters, Model size, Latency (ms/clip), Throughput (clips/s), Peak VRAM / Peak RAM
  Cả pipeline          Total parameters, Latency (ms/frame), Throughput (FPS), Combined peak VRAM / RAM

Dùng đúng model + tham số của web (configs/web.yaml, mục classifier.choices).
Đo trên các khung của MỘT video (giải mã sẵn vào RAM, không tính thời gian đọc file), để các lần đo so được với nhau.

  python bench_report.py                                     # GPU (nếu có), video mặc định
  python bench_report.py --cpu                               # ép CPU (gần điều kiện thiết bị biên hơn)
  python bench_report.py --checkpoint models/b1_lstm.pt      # bộ phân loại khác
  python bench_report.py --source data/videos/fall/sub4_11.mp4 --frames 500

Ra: output/bench/report_<label>_<thời gian>.md / .tex / .json

Cách tính:
  Latency      trung bình trên --frames khung (pose, cả pipeline) hoặc --clips cửa sổ (phân loại), batch = 1,
               sau --warmup lần chạy khởi động. Có kèm p95.
  Throughput   1000 / latency trung bình (xử lý tuần tự, batch = 1).
  Peak VRAM    torch.cuda.max_memory_allocated() trong lúc chạy thành phần đó, TRỪ phần đã cấp phát trước khi nạp
               nó (gồm trọng số + bộ nhớ tạm). Cả pipeline: tính từ trước khi nạp model nào.
  Peak RAM     RSS lớn nhất của tiến trình (lấy mẫu mỗi khung) TRỪ RSS trước khi nạp thành phần đó.
               Thư viện (torch, ultralytics, CUDA) được import TRƯỚC khi lấy mốc nên không bị tính.
  Model size   dung lượng trọng số FP32 = số tham số x 4 byte (kèm dung lượng file .pt trong JSON).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402
import torch  # noqa: E402  (import trước khi lấy mốc RAM: không tính thư viện vào model)
import ultralytics  # noqa: E402,F401

from web.config import load_config, resolve  # noqa: E402
from web.engine import build_classifier, build_estimator, build_filter  # noqa: E402
from src.features import normalize_clip  # noqa: E402
from src.skeleton_io import from_poses  # noqa: E402

PROC = psutil.Process()


class Meter:
    """Theo dõi peak RAM (RSS lấy mẫu) và peak VRAM tính từ một mốc."""

    def __init__(self, cuda: bool):
        self.cuda = cuda

    def mark(self):
        """Mốc TRƯỚC khi nạp / chạy thành phần."""
        if self.cuda:
            torch.cuda.synchronize()
        self.rss0 = PROC.memory_info().rss
        self.rss_peak = self.rss0
        self.vram0 = torch.cuda.memory_allocated() if self.cuda else 0

    def start_peak(self):
        """Bắt đầu đo đỉnh (sau warm-up)."""
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def sample(self):
        self.rss_peak = max(self.rss_peak, PROC.memory_info().rss)

    def result(self) -> dict:
        out = dict(peak_ram_mb=round((self.rss_peak - self.rss0) / 2**20, 1))
        if self.cuda:
            torch.cuda.synchronize()
            out["peak_vram_mb"] = round((torch.cuda.max_memory_allocated() - self.vram0) / 2**20, 1)
        return out


def sync(cuda):
    if cuda:
        torch.cuda.synchronize()


def lat_stats(ms) -> dict:
    ms = np.asarray(ms, dtype=float)
    return dict(latency_ms=round(float(ms.mean()), 2), latency_p95_ms=round(float(np.percentile(ms, 95)), 2),
                throughput=round(1000.0 / float(ms.mean()), 1), n=int(len(ms)))


def load_frames(spec: str, n: int, target_fps: float | None) -> list:
    import cv2
    path = Path(spec) if Path(spec).exists() else resolve(spec)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video {spec}")
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30.0) / target_fps)) if target_fps else 1
    frames, k = [], 0
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if k % step == 0:
            frames.append(f)
        k += 1
    cap.release()
    if not frames:
        raise SystemExit(f"Video {spec} không có khung nào")
    base = list(frames)
    while len(frames) < n:                         # video ngắn: lặp lại cho đủ số khung
        frames += base[: n - len(frames)]
    return frames


def n_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def mb(x_bytes) -> float:
    return round(x_bytes / 2**20, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/videos/fall/sub3_02.mp4", help="video dùng để đo")
    ap.add_argument("--frames", type=int, default=300, help="số khung đo (sau warm-up)")
    ap.add_argument("--clips", type=int, default=500, help="số cửa sổ đo cho bộ phân loại")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--checkpoint", default=None, help="bộ phân loại (mặc định: classifier.checkpoint)")
    ap.add_argument("--cpu", action="store_true", help="ép chạy CPU")
    ap.add_argument("--label", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.checkpoint:
        cfg["classifier"]["checkpoint"] = args.checkpoint
    if args.cpu:
        cfg["pose"]["device"] = "cpu"
        cfg["classifier"]["device"] = "cpu"
    cuda = (not args.cpu) and torch.cuda.is_available()
    if cuda:
        torch.cuda.init()                          # khởi tạo CUDA + nạp cuDNN trước mốc: thư viện không tính vào model
        x = torch.zeros(1, 3, 32, 32, device="cuda")
        torch.nn.functional.conv2d(x, torch.zeros(4, 3, 3, 3, device="cuda"))
        torch.nn.functional.conv2d(x.half(), torch.zeros(4, 3, 3, 3, device="cuda").half())
        del x
        torch.cuda.synchronize()
    meter = Meter(cuda)
    W_total = args.warmup + args.frames
    frames = load_frames(args.source, W_total, cfg["source"].get("target_fps"))
    print(f"Đã nạp {len(frames)} khung từ {args.source} | thiết bị: {'cuda' if cuda else 'cpu'}")

    # ------------------------------------------------------------ 1. Pose
    meter.mark()
    base_rss, base_vram = meter.rss0, meter.vram0      # mốc trước khi nạp model nào
    est = build_estimator(cfg)
    yolo = est.backend.model
    pose_path = Path(est.backend.model_path)
    for i in range(args.warmup):
        est.process_frame(frames[i], i, i / 10)
    meter.start_peak()
    pose_ms, poses = [], []
    for i in range(args.warmup, W_total):
        t = time.perf_counter()
        p = est.process_frame(frames[i], i, i / 10)
        sync(cuda)
        pose_ms.append((time.perf_counter() - t) * 1000)
        meter.sample()
        poses.append(p)
    pose = dict(name=pose_path.name, params=n_params(yolo.model), file_mb=mb(pose_path.stat().st_size),
                imgsz=cfg["pose"].get("imgsz"), **lat_stats(pose_ms), **meter.result())
    pose["size_mb"] = mb(pose["params"] * 4)
    print(f"Pose: {pose['latency_ms']} ms/khung")

    # ------------------------------------------------------------ 2. Bộ phân loại
    flt = build_filter(cfg)
    if flt is not None:
        poses = [flt.smooth(p) for p in poses]
    meter.mark()
    clf = build_classifier(cfg)
    if clf is None:
        raise SystemExit("Chưa có model phân loại (classifier.checkpoint).")
    W = clf.window_size
    item = from_poses(poses)
    clips = [normalize_clip(item["seq"][e - W + 1:e + 1], item["mask"][e - W + 1:e + 1], item["aspect"])
             for e in range(W - 1, len(poses))] or [np.zeros((W, 17, 3), np.float32)]
    for i in range(args.warmup):
        clf.predict_batch(clips[i % len(clips)][None])
    meter.start_peak()
    clf_ms = []
    for i in range(args.clips):
        x = clips[i % len(clips)][None]
        t = time.perf_counter()
        clf.predict_batch(x)                       # đã .cpu() bên trong -> đồng bộ GPU
        clf_ms.append((time.perf_counter() - t) * 1000)
        meter.sample()
    ck = Path(clf.checkpoint_path)
    classifier = dict(name=clf.name, arch=clf.arch, params=clf.num_params, file_mb=mb(ck.stat().st_size),
                      window=W, threshold=clf.fall_threshold, postprocess=clf.post is not None,
                      **lat_stats(clf_ms), **meter.result())
    classifier["size_mb"] = mb(classifier["params"] * 4)
    print(f"Phân loại: {classifier['latency_ms']} ms/cửa sổ")

    # ------------------------------------------------------------ 3. Cả pipeline (pose -> lọc -> cửa sổ -> model -> hậu xử lý)
    # Dùng lại 2 model đã nạp; mốc bộ nhớ = lúc CHƯA nạp model nào (base) -> gồm trọng số cả hai + bộ nhớ tạm.
    meter.rss0, meter.vram0 = base_rss, base_vram
    meter.rss_peak = PROC.memory_info().rss
    est.reset()
    clf.reset()
    if flt is not None:
        flt.reset()
    meter.start_peak()
    e2e_ms = []
    for k, i in enumerate(range(args.warmup, W_total)):
        t = time.perf_counter()
        p = est.process_frame(frames[i], k, k / 10)
        if flt is not None:
            p = flt.smooth(p)
        clf.update(p)
        sync(cuda)
        e2e_ms.append((time.perf_counter() - t) * 1000)
        meter.sample()
    pipe = dict(params=pose["params"] + classifier["params"], **lat_stats(e2e_ms), **meter.result())
    est.close()

    # ------------------------------------------------------------ xuất bảng
    dev = torch.cuda.get_device_name(0) if cuda else (psutil.cpu_count(logical=False) and
                                                      f"CPU ({psutil.cpu_count()} luồng)")
    mem_key, mem_name = ("peak_vram_mb", "Peak VRAM") if cuda else ("peak_ram_mb", "Peak RAM")
    pose_title = Path(pose["name"]).stem
    clf_title = f"{classifier['arch']} ({Path(classifier['name'].split(':')[-1]).name}, window {W})"
    rows = [
        (f"Pose: {pose_title}", None),
        ("Parameters", f"{pose['params']:,}"),
        ("Model size (FP32)", f"{pose['size_mb']:.2f} MB"),
        ("Latency", f"{pose['latency_ms']:.2f} ms/frame"),
        ("Throughput", f"{pose['throughput']:.1f} FPS"),
        (mem_name, f"{pose[mem_key]:.1f} MB"),
        (f"Classifier: {clf_title}", None),
        ("Parameters", f"{classifier['params']:,}"),
        ("Model size (FP32)", f"{classifier['size_mb']:.2f} MB"),
        ("Latency", f"{classifier['latency_ms']:.2f} ms/clip"),
        ("Throughput", f"{classifier['throughput']:.0f} clips/sec"),
        (mem_name, f"{classifier[mem_key]:.1f} MB"),
        ("Combined Pipeline", None),
        ("Total Parameters", f"{pipe['params'] / 1e6:.2f}M"),
        ("End-to-end latency", f"{pipe['latency_ms']:.2f} ms/frame"),
        ("End-to-end throughput", f"{pipe['throughput']:.1f} FPS"),
        (f"Combined {mem_name}", f"{pipe[mem_key]:.1f} MB"),
    ]
    if cuda:                                       # GPU: thêm RAM của cả pipeline (CPU) cho đủ
        rows.append(("Combined Peak RAM", f"{pipe['peak_ram_mb']:.1f} MB"))

    width = max(len(r[0]) for r in rows) + 2
    print("\n" + "=" * (width + 22))
    print(f"{'Metric':{width}s}{'Value':>20s}")
    print("-" * (width + 22))
    for name, val in rows:
        print(name if val is None else f"  {name:{width - 2}s}{val:>20s}")
    print("=" * (width + 22))
    print(f"Thiết bị: {dev} | video: {args.source} | {args.frames} khung, {args.clips} cửa sổ, batch 1")

    md = ["| Metric | Value |", "|---|---:|"]
    for name, val in rows:
        md.append(f"| **{name}** | |" if val is None else f"| {name} | {val} |")
    md.append(f"\n*Thiết bị: {dev}; {args.frames} khung từ `{args.source}`, batch 1, sau {args.warmup} lần warm-up.*")

    def tex(s):
        return s.replace("_", r"\_").replace("%", r"\%")
    tx = [r"\begin{tabular}{lr}", r"\toprule", r"\textbf{Metric} & \textbf{Value} \\", r"\midrule"]
    first = True
    for name, val in rows:
        if val is None:
            if not first:
                tx.append(r"\midrule")
            tx.append(rf"\multicolumn{{2}}{{l}}{{\textit{{{tex(name)}}}}} \\")
            first = False
        else:
            tx.append(rf"{tex(name)} & {tex(val)} \\")
    tx += [r"\bottomrule", r"\end{tabular}"]

    label = args.label or ("gpu" if cuda else "cpu") + "_" + classifier["arch"]
    out = resolve("output/bench")
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"report_{label}_{datetime.now():%Y%m%d_%H%M%S}"
    stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    stem.with_suffix(".tex").write_text("\n".join(tx) + "\n", encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(dict(
        device=dev, cuda=cuda, source=args.source, frames=args.frames, clips=args.clips, warmup=args.warmup,
        pose=pose, classifier=classifier, pipeline=pipe, time=datetime.now().isoformat(timespec="seconds")),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu: {stem}.md / .tex / .json")


if __name__ == "__main__":
    main()
