"""
pipeline.py
-----------
Toàn bộ logic chạy pipeline: nguồn -> PoseEstimate -> SkeletonFilter
-> CSV (+ ảnh / video overlay). main.py chỉ tạo PipelineConfig và gọi run().
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .csv_export import CSVExporter
from .pose_estimate import PoseEstimate
from .postprocess import interpolate_csv
from .skeleton_draw import SkeletonDraw
from .skeleton_filter import SkeletonFilter
from .video_process import FileSource
from .webcam import StreamSource

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


@dataclass
class PipelineConfig:
    mode: str = "video"                  # "video" | "webcam"
    video_path: str = ""                 # file hoặc thư mục video
    model_path: str = "model/yolo11n-pose.pt"
    webcam_source: int | str = 0
    webcam_max_seconds: float | None = None
    output_dir: str = "output"
    target_fps: float = 10.0
    conf: float = 0.4
    device: str | None = None
    use_filter: bool = True
    max_frames: int | None = None
    save_frames: bool = True
    frames_with_pose: bool = True
    show_video: bool = False
    save_video_video: bool = True
    show_webcam: bool = True
    save_webcam_video: bool = False
    # độ chính xác pose
    nms_iou: float = 0.5
    kp_conf: float = 0.35
    use_tracker: bool = True
    tracker: str = "bytetrack.yaml"
    imgsz: int | None = None
    # bộ lọc
    max_hold_frames: int = 5
    min_cutoff: float = 1.5
    beta: float = 4.0
    # nội suy offline trên CSV
    interpolate: bool = True
    interp_max_gap: int = 6
    # vẽ
    show_bbox: bool = True
    show_score: bool = True
    show_kp_conf: bool = False


def list_videos(path):
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
    if p.is_file():
        return [p]
    sys.exit(f"Không tìm thấy video_path: {path}")


def process_source(src, estimator, flt, drawer, cfg, show, save_video):
    """Chạy pipeline cho một nguồn. Trả True nếu người dùng nhấn q để thoát hẳn."""
    estimator.reset()
    if flt is not None:
        flt.reset()

    writer = None
    quit_all = False
    print(f"\n== {src.video_id} | out_fps={src.out_fps:.1f} ==")

    with CSVExporter(src.video_id, cfg.output_dir) as csv_out:
        try:
            for pkt in src.get_frames():
                pose = estimator.process_frame(pkt.frame, pkt.idx, pkt.t)
                if flt is not None:
                    pose = flt.smooth(pose)
                csv_out.write(pose)

                need_vis = show or save_video or (cfg.save_frames and cfg.frames_with_pose)
                vis = drawer.draw(pkt.frame, pose) if need_vis else None

                if cfg.save_frames:
                    src.save_frame(vis if cfg.frames_with_pose else pkt.frame, pkt.idx)

                if show or save_video:
                    if save_video:
                        if writer is None:
                            out = Path(cfg.output_dir) / "video" / f"{src.video_id}_overlay.mp4"
                            out.parent.mkdir(parents=True, exist_ok=True)
                            h, w = vis.shape[:2]
                            writer = cv2.VideoWriter(
                                str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                src.out_fps, (w, h))
                        writer.write(vis)
                    if show:
                        cv2.imshow("pipeline (q de thoat)", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            quit_all = True
                            break
        except KeyboardInterrupt:
            quit_all = True
            print("\nĐã dừng bằng Ctrl+C.")
        finally:
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        print(f"Số khung: {csv_out.total} | valid=0: {csv_out.invalid} | "
              f"missing_ratio: {csv_out.missing_ratio:.3f}")
        print(f"CSV: {csv_out.output_path}")

    if cfg.interpolate:
        out, n_filled = interpolate_csv(csv_out.output_path, cfg.interp_max_gap)
        print(f"CSV nội suy: {out} ({n_filled} ô khớp được lấp)")
    return quit_all


def run_video_mode(estimator, flt, drawer, cfg):
    videos = list_videos(cfg.video_path)
    if not videos:
        sys.exit(f"Không có video nào trong: {cfg.video_path}")
    for video in videos:
        try:
            src = FileSource(video, cfg.output_dir, cfg.target_fps, cfg.max_frames)
        except ValueError as e:
            print(f"Bỏ qua {video}: {e}")
            continue
        with src:
            if process_source(src, estimator, flt, drawer, cfg,
                              cfg.show_video, cfg.save_video_video):
                break


def run_webcam_mode(estimator, flt, drawer, cfg):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"webcam{cfg.webcam_source}" if isinstance(cfg.webcam_source, int) else "stream"
    try:
        src = StreamSource(
            cfg.webcam_source, cfg.output_dir, cfg.target_fps,
            max_frames=cfg.max_frames, max_seconds=cfg.webcam_max_seconds,
            video_id=f"{name}_{stamp}",
        )
    except RuntimeError as e:
        sys.exit(f"Lỗi camera: {e}")
    with src:
        process_source(src, estimator, flt, drawer, cfg,
                       cfg.show_webcam, cfg.save_webcam_video)


def run(cfg: PipelineConfig):
    if cfg.mode not in ("video", "webcam"):
        sys.exit('mode phải là "video" hoặc "webcam".')

    t0 = time.time()
    flt = (
        SkeletonFilter(
            min_conf=cfg.kp_conf,
            max_hold_frames=cfg.max_hold_frames,
            min_cutoff=cfg.min_cutoff,
            beta=cfg.beta,
        )
        if cfg.use_filter else None
    )
    drawer = SkeletonDraw(
        min_conf=cfg.kp_conf,
        show_bbox=cfg.show_bbox,
        show_score=cfg.show_score,
        show_kp_conf=cfg.show_kp_conf,
    )

    with PoseEstimate(
        cfg.model_path, conf=cfg.conf, device=cfg.device, kp_conf=cfg.kp_conf,
        iou=cfg.nms_iou, use_tracker=cfg.use_tracker, tracker=cfg.tracker,
        imgsz=cfg.imgsz,
    ) as estimator:
        if cfg.mode == "video":
            run_video_mode(estimator, flt, drawer, cfg)
        else:
            run_webcam_mode(estimator, flt, drawer, cfg)

    print(f"\nHoàn tất sau {time.time() - t0:.1f}s. Kết quả trong '{cfg.output_dir}/'.")