"""
extract_skeleton.py
-------------------
Video -> output/npz/<video_id>.npz  (đầu vào của build_dataset.py).

Mỗi video: YOLO-pose -> SkeletonFilter -> list[PoseResult] -> from_poses -> NPZ.
NPZ chứa: seq, mask, valid, t, frame_idx, fps, aspect (đúng khóa build_dataset cần).

video_id = "<subject>_<kind>_<tên tệp>", ví dụ "Subject 1_Fall_01".
build_dataset nhận video ngã khi kind = "Fall", và lấy phần trước dấu "_" đầu
tiên làm subject để chia train/val/test.

Chạy (mỗi lần một thư mục video của MỘT người, MỘT loại):
    python extract_skeleton.py data/videos/s1_fall --subject "Subject 1" --kind Fall
    python extract_skeleton.py data/videos/s1_adl  --subject "Subject 1" --kind ADL

Đặt tệp này ở thư mục gốc, ngang hàng với package src/.
(fall_web: model mặc định đổi từ model/yolo11n-pose.pt thành tên trần yolo11n-pose.pt.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.pose_estimate import PoseEstimate
from src.skeleton_filter import SkeletonFilter
from src.skeleton_io import from_poses
from src.video_process import FileSource

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def parse_imgsz(text):
    """'224,320' -> (224, 320) (cao, rộng); '0' hoặc 'none' -> None (mặc định model)."""
    if text is None or text.strip().lower() in ("0", "none", ""):
        return None
    h, w = (int(v) for v in text.split(","))
    return (h, w)


def extract_video(video, video_id, estimator, flt, out_dir, fps, max_frames=None):
    estimator.reset()
    flt.reset()
    poses = []
    with FileSource(video, out_dir, fps, max_frames, video_id=video_id) as src:
        for pkt in src.get_frames():
            pose = estimator.process_frame(pkt.frame, pkt.idx, pkt.t)
            poses.append(flt.smooth(pose))
        out_fps = src.out_fps
    if not poses:
        print(f"[bỏ qua] {video_id}: không đọc được khung nào")
        return None

    item = from_poses(poses)
    out = Path(out_dir) / "npz" / f"{video_id}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **item)

    n_valid = int(item["valid"].sum())
    n_obs = int(item["mask"].sum())
    print(f"{video_id}: {item['seq'].shape} | fps={item['fps']:.1f} (nguồn {out_fps:.1f}) | "
          f"khung có người {n_valid}/{len(poses)} | khớp quan sát {n_obs / item['mask'].size:.0%}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir", help="thư mục video (hoặc một tệp video)")
    ap.add_argument("--subject", required=True, help='ví dụ "Subject 1"')
    ap.add_argument("--kind", required=True, help='"Fall" hoặc tên loại ADL (Walk, Sit...)')
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="tên trần thì ultralytics tự tải; hoặc models/yolo11n-pose.pt")
    ap.add_argument("--out", default="output")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="ngưỡng phát hiện người; trùng conf_th trên board")
    ap.add_argument("--kp-conf", type=float, default=0.25,
                    help="ngưỡng khớp; trùng keypoint_th trên board")
    ap.add_argument("--imgsz", default="224,320",
                    help='"cao,rộng" khớp MaixCAM; "0" = mặc định của model (640)')
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    if "_" in args.subject or "_" in args.kind:
        sys.exit('subject và kind không được chứa dấu "_" (build_dataset tách id theo "_").')

    p = Path(args.video_dir)
    videos = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS) if p.is_dir() else [p]
    if not videos:
        sys.exit(f"Không có video trong {p}")

    flt = SkeletonFilter(min_conf=args.kp_conf)
    with PoseEstimate(args.model, conf=args.conf, device=args.device,
                      kp_conf=args.kp_conf, imgsz=parse_imgsz(args.imgsz)) as est:
        for v in videos:
            vid = f"{args.subject}_{args.kind}_{v.stem}"
            try:
                extract_video(v, vid, est, flt, args.out, args.fps, args.max_frames)
            except ValueError as e:
                print(f"[bỏ qua] {v}: {e}")


if __name__ == "__main__":
    main()