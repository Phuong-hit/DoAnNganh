"""render_overlay.py
-----------------
Xuất video MP4 có vẽ khung xương (skeleton overlay) và trạng thái phát hiện ngã
để kiểm tra trực quan chất lượng bám khớp và cảnh báo.

Chạy:
    python render_overlay.py --video data/raw/07.mp4
    python render_overlay.py --video data/raw/01.mp4 --model models/yolo11n-pose.pt
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Thêm thư mục gốc vào path để import src.*
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.b0_rule import B0Rule, OnlineFallDetector
from src.data_utils import load_model_json
from src.pose_estimate import PoseEstimate
from src.skeleton_draw import SkeletonDraw
from src.skeleton_filter import SkeletonFilter


def render_overlay(
    video_path: str,
    output_path: str | None = None,
    model_path: str = "models/yolo11n-pose.pt",
    conf: float = 0.35,
    use_filter: bool = True,
    fps_sample: float | None = None,
):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    if output_path is None:
        out_dir = Path("output") / "overlays"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{video_path.stem}_overlay.mp4"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = float(fps_sample) if fps_sample else src_fps

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))

    print(f"Bắt đầu xử lý: {video_path.name}")
    print(f"Kích thước: {width}x{height} | FPS gốc: {src_fps:.1f} | Tổng khung: {total_frames}")
    print(f"Đầu ra: {output_path}")

    # Khởi tạo pipeline
    estimator = PoseEstimate(model_path=model_path, conf=conf)
    flt = SkeletonFilter() if use_filter else None
    drawer = SkeletonDraw(min_conf=0.3, show_status=True, show_bbox=True, show_score=True)

    # Nạp luật B0 cải tiến
    try:
        rule_params = B0Rule.from_params(load_model_json("b0_rule")["params"]).params
    except Exception:
        from src.b0_rule import B0Params
        rule_params = B0Params()

    detector = OnlineFallDetector(window_size=20, fps=10.0, params=rule_params, latch_seconds=3.0)

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / src_fps
        pose = estimator.process_frame(frame, frame_idx, t)

        if flt is not None:
            pose = flt.smooth(pose)

        # Cập nhật bộ phát hiện ngã
        state = detector.update(pose.joints, aspect=pose.aspect)

        # Vẽ skeleton
        vis_frame = drawer.draw(frame, pose)

        # Vẽ HUD trạng thái ngã
        if state["is_fall"]:
            txt = "CANH BAO: NGA KHUAT!" if state["is_occluded"] else "CANH BAO: PHAT HIEN NGA!"
            cv2.rectangle(vis_frame, (10, height - 70), (width - 10, height - 15), (0, 0, 200), -1)
            cv2.putText(vis_frame, txt, (25, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        else:
            cv2.putText(vis_frame, "TRANG THAI: BINH THUONG", (20, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Thêm thông tin khung hình
        cv2.putText(vis_frame, f"Frame: {frame_idx}/{total_frames} (t={t:.2f}s)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        writer.write(vis_frame)
        frame_idx += 1

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"  -> Đã xử lý {frame_idx}/{total_frames} khung hình...")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"Xong! Lưu video tại: {output_path} ({elapsed:.1f}s)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render video MP4 có skeleton overlay.")
    parser.add_argument("--video", default="data/raw/07.mp4", help="Đường dẫn file video đầu vào")
    parser.add_argument("--output", default=None, help="Đường dẫn file video đầu ra (.mp4)")
    parser.add_argument("--model", default="models/yolo11n-pose.pt", help="Đường dẫn model YOLO-pose")
    parser.add_argument("--conf", type=float, default=0.35, help="Ngưỡng confidence phát hiện người")
    parser.add_argument("--no-filter", action="store_true", help="Tắt bộ lọc SkeletonFilter")
    args = parser.parse_args()

    render_overlay(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        conf=args.conf,
        use_filter=not args.no_filter,
    )