"""
extract_skeleton.py
-------------------
Video -> output/npz/<video_id>.npz  (đầu vào của build_dataset.py).

Mỗi video: YOLO-pose -> SkeletonFilter -> list[PoseResult] -> from_poses -> NPZ.
NPZ chứa: seq, mask, valid, t, frame_idx, fps, aspect (đúng khóa build_dataset cần).

video_id = "<subject>_<kind>_<mã video>", ví dụ "SBJ01_ADL_LOC3-ACT13-B-1".
build_dataset nhận video ngã khi kind = "Fall", và lấy phần trước dấu "_" đầu
tiên làm subject để chia train/val/test. Vì vậy subject và kind không được chứa
dấu "_"; phần mã video thì "_" đã được đổi thành "-".

HAI CÁCH CHẠY

1) Cả bộ dữ liệu có cấu trúc  <gốc>/<ADL|Fall>/SBJ_01_LOC3/ACT13_B_1/xxx.mp4
   (quét mọi thư mục con; subject, kind, mã video lấy từ đường dẫn):
       python extract_skeleton.py "D:\\dataset\\VideoDataset" --dry-run   # xem tên trước, chưa chạy YOLO
       python extract_skeleton.py "D:\\dataset\\VideoDataset"
       python extract_skeleton.py "D:\\dataset\\VideoDataset" --skip-existing   # chạy tiếp sau khi bị ngắt
   Cũng chạy được trên một nhánh: ...\\VideoDataset\\Fall  hoặc ...\\ADL\\SBJ_01_LOC3

2) Thư mục video phẳng của MỘT người, MỘT loại (như cũ):
       python extract_skeleton.py data/videos/s1_fall --subject "Subject 1" --kind Fall

Đặt tệp này ở thư mục gốc, ngang hàng với package src/.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.pose_estimate import PoseEstimate
from src.skeleton_filter import SkeletonFilter
from src.skeleton_io import from_poses
from src.video_process import FileSource

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
KIND_NAMES = {"adl": "ADL", "fall": "Fall"}
SUBJECT_RE = re.compile(r"^SBJ[_-]?(\d+)(?:[_-]?(LOC\d+))?$", re.IGNORECASE)


def parse_imgsz(text):
    """'224,320' -> (224, 320) (cao, rộng); '0' hoặc 'none' -> None (mặc định model)."""
    if text is None or text.strip().lower() in ("0", "none", ""):
        return None
    h, w = (int(v) for v in text.split(","))
    return (h, w)


def clean(text):
    """Bỏ dấu '_' và khoảng trắng để không làm vỡ cách tách tên của build_dataset."""
    return re.sub(r"[_\s]+", "-", str(text)).strip("-")


def find_videos(root):
    """Một tệp, hoặc mọi video trong thư mục và các thư mục con (bỏ qua .txt, .json...)."""
    p = Path(root)
    if p.is_file():
        return [p]
    if not p.is_dir():
        sys.exit(f"Không tìm thấy: {p}")
    return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTS)


def parse_dataset_path(video):
    """.../ADL/SBJ_01_LOC3/ACT13_B_1/xxx.mp4 -> (subject, kind, tag) hoặc None nếu không khớp.

    subject = "SBJ01"; kind = "ADL" | "Fall"; tag = "LOC3-ACT13-B-1".
    """
    parts = Path(video).parts[:-1]
    kind = next((KIND_NAMES[p.lower()] for p in parts if p.lower() in KIND_NAMES), None)
    for i, p in enumerate(parts):
        m = SUBJECT_RE.match(p)
        if m:
            break
    else:
        return None
    if kind is None:
        return None
    tag = [m[2].upper()] if m[2] else []
    tag += [clean(q) for q in parts[i + 1:]]            # thư mục con sau SBJ_xx_LOCx (ACT13_B_1)
    return f"SBJ{m[1]}", kind, "-".join(tag) or clean(Path(video).stem)


def plan_jobs(videos, subject=None, kind=None):
    """Trả (jobs [(đường dẫn, video_id)], danh sách video không đặt được tên)."""
    jobs, unmatched = [], []
    for v in videos:
        if subject and kind:
            jobs.append((v, f"{subject}_{kind}_{clean(v.stem)}"))
            continue
        info = parse_dataset_path(v)
        if info is None:
            unmatched.append(v)
        else:
            jobs.append((v, "_".join(info)))
    # cùng một thư mục có nhiều video -> thêm tên tệp để id không trùng
    count = Counter(vid for _, vid in jobs)
    jobs = [(v, vid if count[vid] == 1 else f"{vid}-{clean(v.stem)}") for v, vid in jobs]
    return jobs, unmatched


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
    ap.add_argument("video_dir", help="thư mục gốc của bộ dữ liệu (hoặc một thư mục / một tệp video)")
    ap.add_argument("--subject", default=None,
                    help='chỉ dùng với thư mục phẳng, ví dụ "Subject 1" (bỏ trống = lấy từ đường dẫn)')
    ap.add_argument("--kind", default=None,
                    help='chỉ dùng với thư mục phẳng: "Fall" hoặc loại ADL (bỏ trống = lấy từ đường dẫn)')
    ap.add_argument("--model", default="model/yolo26m-pose.pt")
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
    ap.add_argument("--dry-run", action="store_true",
                    help="chỉ in tên NPZ sẽ tạo cho từng video, không chạy YOLO")
    ap.add_argument("--skip-existing", action="store_true",
                    help="bỏ qua video đã có NPZ (chạy tiếp sau khi bị ngắt)")
    args = ap.parse_args()

    if bool(args.subject) != bool(args.kind):
        sys.exit("--subject và --kind phải đi cùng nhau (hoặc bỏ cả hai để lấy từ đường dẫn).")
    for name in (args.subject, args.kind):
        if name and "_" in name:
            sys.exit('subject và kind không được chứa dấu "_" (build_dataset tách id theo "_").')

    videos = find_videos(args.video_dir)
    if not videos:
        sys.exit(f"Không có video trong {args.video_dir}")
    jobs, unmatched = plan_jobs(videos, args.subject, args.kind)

    print(f"Tìm thấy {len(videos)} video | đặt tên được {len(jobs)} | không đặt được {len(unmatched)}")
    for v in unmatched[:5]:
        print(f"  [không khớp cấu trúc .../ADL|Fall/SBJ_xx_LOCx/...] {v}")
    if len(unmatched) > 5:
        print(f"  ... và {len(unmatched) - 5} video nữa")
    if not jobs:
        sys.exit("Không có video nào đặt được tên. Kiểm tra cấu trúc thư mục, "
                 "hoặc dùng --subject và --kind cho thư mục phẳng.")

    by_kind = Counter(vid.split("_")[1] for _, vid in jobs)
    by_subject = Counter(vid.split("_")[0] for _, vid in jobs)
    print("Theo loại:", dict(by_kind), "| số subject:", len(by_subject))

    if args.dry_run:
        for v, vid in jobs:
            print(f"{vid}   <-   {v}")
        return

    flt = SkeletonFilter(min_conf=args.kp_conf)
    skipped, failed = 0, []
    with PoseEstimate(args.model, conf=args.conf, device=args.device,
                      kp_conf=args.kp_conf, imgsz=parse_imgsz(args.imgsz)) as est:
        for i, (v, vid) in enumerate(jobs, 1):
            if args.skip_existing and (Path(args.out) / "npz" / f"{vid}.npz").exists():
                skipped += 1
                continue
            print(f"[{i}/{len(jobs)}] ", end="")
            try:
                extract_video(v, vid, est, flt, args.out, args.fps, args.max_frames)
            except Exception as e:                       # một video hỏng không làm dừng cả lượt
                failed.append((v, e))
                print(f"[lỗi] {v}: {type(e).__name__}: {e}")

    n_npz = len(list((Path(args.out) / "npz").glob("*.npz")))
    print(f"\nXong. Bỏ qua (đã có): {skipped} | lỗi: {len(failed)} | tổng NPZ trong {args.out}/npz: {n_npz}")
    for v, e in failed:
        print(f"  lỗi: {v} -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()