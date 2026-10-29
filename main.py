#!/usr/bin/env python3
"""
main.py
-------
Chạy phát hiện ngã trên video / webcam / luồng RTSP, hiển thị bằng cửa sổ OpenCV (không cần web):

    Video/Webcam -> YOLO-Pose -> SkeletonFilter -> FallClassifier (ST-GCN hoặc LSTM) -> NGÃ / KHÔNG NGÃ

Thay main.py của code_1. Khác bản cũ:
  - dùng ĐÚNG pipeline đã tạo dữ liệu huấn luyện (FileSource/StreamSource + PoseEstimate + SkeletonFilter của src/,
    cấu hình configs/code1.yaml = tham số của extract_skeleton.py), nên skeleton lúc chạy giống lúc train.
    Bản cũ tự lấy pose bằng predict() không lọc, không imgsz, nên khác dữ liệu huấn luyện;
  - chọn model bằng --checkpoint (ST-GCN hay LSTM do checkpoint quyết định);
  - không còn đường dẫn D:/... cứng; in các đoạn bị báo ngã và có thể lưu P(ngã) theo thời gian.

Ví dụ (từ thư mục gốc project):
  python main.py --source "D:/dataset/Subject 2/FALL/02.mp4"
  python main.py --source "D:/dataset/Subject 2/FALL/02.mp4" --checkpoint models/best_lstm.pt
  python main.py --source 0                                   # webcam
  python main.py --source video.mp4 --save output/prediction.mp4 --log output/prediction.csv --no-show
Nhấn q trên cửa sổ hình để thoát.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from web.config import load_config, resolve  # noqa: E402
from web.engine import (OpenedSource, build_classifier, build_drawer, build_estimator,  # noqa: E402
                        build_filter, draw_fall, draw_hud, estimate, pose_tag)


def fall_segments(rows, threshold):
    """[(t, p)] -> các đoạn [t_bắt_đầu, t_kết_thúc] liên tiếp có p >= ngưỡng."""
    segs, start, last = [], None, None
    for t, p in rows:
        if p >= threshold and start is None:
            start = t
        if p < threshold and start is not None:
            segs.append((start, last))
            start = None
        last = t
    if start is not None:
        segs.append((start, last))
    return segs


def run(args) -> int:
    overrides = list(args.set)
    if args.checkpoint:
        overrides.append(f"classifier.checkpoint={args.checkpoint}")
    if args.threshold is not None:
        overrides.append(f"classifier.threshold={args.threshold}")
    if args.no_filter:
        overrides.append("filter.enabled=false")
    cfg = load_config(resolve(args.config), overrides)

    clf = build_classifier(cfg)
    if clf is None:
        sys.exit("Chưa có checkpoint: dùng --checkpoint models/best_stgcn.pt (hoặc best_lstm.pt)")
    print(f"Bộ phân loại: {clf.name} | {clf.num_params:,} tham số | cửa sổ {clf.window_size} khung, "
          f"dự đoán mỗi {clf.stride_frames} khung | ngưỡng {clf.fall_threshold:.2f} | {clf.device}")

    estimator = build_estimator(cfg)
    flt = build_filter(cfg)
    drawer = build_drawer(cfg)
    writer = None
    log_rows: list[tuple[float, float]] = []
    n = 0
    t_wall = time.perf_counter()
    try:
        with OpenedSource(args.source, cfg) as src:
            print(f"Nguồn: {src.video_id} ({src.kind}, {src.out_fps:g} fps sau lấy mẫu)")
            estimator.reset()
            if flt is not None:
                flt.reset()
            clf.reset()
            for pkt in src.frames():
                pose, lat = estimate(estimator, flt, pkt.frame, pkt.idx, pkt.t)
                res = clf.update(pose)
                if res is not None:
                    log_rows.append((round(pkt.t, 3), round(res["probability"], 4)))
                n += 1
                img = drawer.draw(pkt.frame, pose)
                draw_fall(img, clf.last)
                fps = n / max(1e-6, time.perf_counter() - t_wall)
                draw_hud(img, [f"{pose_tag(cfg)} + {clf.name}  t={pkt.t:.1f}s",
                               f"pose {lat:.0f} ms | {fps:.1f} fps"])
                if args.save:
                    if writer is None:
                        out = Path(args.save)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        h, w = img.shape[:2]
                        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), src.out_fps, (w, h))
                    writer.write(img)
                if not args.no_show:
                    cv2.imshow("Fall detection", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        estimator.close()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print(f"\nĐã xử lý {n} khung, {len(log_rows)} lần dự đoán.")
    segs = fall_segments(log_rows, clf.fall_threshold)
    if segs:
        print("Các đoạn bị báo NGÃ (thời điểm cuối cửa sổ 2 s, giây):")
        for a, b in segs:
            print(f"  {a:7.1f} -> {b:7.1f}")
    else:
        print("Không có đoạn nào bị báo ngã.")
    if args.log:
        out = Path(args.log)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "p_fall"])
            w.writerows(log_rows)
        print(f"Đã ghi {out}")
    if args.save:
        print(f"Video kết quả: {args.save}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phát hiện ngã trên video / webcam",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--source", default="0", help="đường dẫn video, số webcam (0) hoặc rtsp://...")
    ap.add_argument("--config", default="configs/code1.yaml")
    ap.add_argument("--checkpoint", default=None, help="mặc định theo classifier.checkpoint trong cấu hình")
    ap.add_argument("--threshold", type=float, default=None, help="mặc định: ngưỡng lưu trong checkpoint")
    ap.add_argument("--no-filter", action="store_true", help="tắt SkeletonFilter")
    ap.add_argument("--save", default=None, help="lưu video kết quả (.mp4)")
    ap.add_argument("--log", default=None, help="lưu P(ngã) theo thời gian (.csv)")
    ap.add_argument("--no-show", action="store_true", help="không mở cửa sổ hình (máy chủ / Colab)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="ghi đè cấu hình")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
