"""
video_process.py
----------------
Nguồn video từ file, lấy mẫu về target_fps (mặc định 10 fps theo quy ước nhóm).

Thời gian: khung thứ k sau lấy mẫu có t = k / out_fps (lưới đều),
frame_idx = k (chỉ số sau lấy mẫu). Chỉ số khung gốc nằm ở src_idx.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FramePacket:
    idx: int          # chỉ số khung sau lấy mẫu
    t: float          # giây
    frame: np.ndarray
    src_idx: int = -1  # chỉ số khung trong nguồn gốc (-1 nếu không có)

    def __iter__(self):  # tương thích: for idx, t, frame in source.get_frames()
        return iter((self.idx, self.t, self.frame))


class FileSource:
    def __init__(
        self,
        source,
        output_dir="output",
        target_fps: float | None = 10.0,
        max_frames: int | None = None,
    ):
        self.source = Path(source)
        self.video_id = self.source.stem
        self.max_frames = max_frames

        self.cap = cv2.VideoCapture(str(self.source))
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError(f"Không thể mở nguồn video: {source}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.src_fps = fps if fps and fps > 0 and math.isfinite(fps) else 30.0
        self.total_src_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps phải > 0 hoặc None.")
        # Không tăng mẫu (upsample): target >= src thì giữ nguyên fps gốc.
        if target_fps is None or target_fps >= self.src_fps * 0.999:
            self.out_fps = self.src_fps
        else:
            self.out_fps = float(target_fps)

        self.frames_dir = Path(output_dir) / "frames" / self.video_id
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def get_frames(self):
        """Yield FramePacket theo lưới thời gian đều out_fps."""
        k = 0
        src_idx = 0
        half_src = 0.5 / self.src_fps
        next_t = 0.0

        while True:
            if self.max_frames is not None and k >= self.max_frames:
                break
            ok, frame = self.cap.read()
            if not ok:
                break

            ts = src_idx / self.src_fps
            # Lấy khung gốc gần nhất với mốc lưới next_t.
            if ts + half_src >= next_t - 1e-9:
                yield FramePacket(k, round(k / self.out_fps, 3), frame, src_idx)
                k += 1
                next_t = k / self.out_fps
            src_idx += 1

    def save_frame(self, frame, frame_idx):
        path = self.frames_dir / f"frame_{frame_idx:06d}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise IOError(f"Không thể lưu frame: {path}")
        return path

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# Tên cũ, giữ để code cũ vẫn chạy.
VideoProcess = FileSource