"""Backend pose GIẢ, cài đặt giao diện PoseBackend của src/pose_estimate.py.

Dùng để chạy thử web / kiểm thử khi chưa cài ultralytics hoặc chưa có model:
    python run_web.py --set pose.backend=dummy

Sinh một người đứng, đi ngang rồi ngã theo thời gian (tính theo số lần gọi detect),
trả về Detection ở đơn vị pixel giống YoloBackend. Không đọc nội dung ảnh.
"""
from __future__ import annotations

import math

import numpy as np

from src.pose_estimate import Detection

# Tư thế đứng chuẩn (x, y) trong hệ tọa độ người: gốc ở giữa hông, đơn vị = chiều cao người.
_STAND = np.array([
    [0.00, -0.55],                  # nose
    [-0.02, -0.57], [0.02, -0.57],  # eyes
    [-0.05, -0.55], [0.05, -0.55],  # ears
    [-0.12, -0.40], [0.12, -0.40],  # shoulders
    [-0.16, -0.20], [0.16, -0.20],  # elbows
    [-0.17, -0.02], [0.17, -0.02],  # wrists
    [-0.08, 0.00], [0.08, 0.00],    # hips
    [-0.09, 0.22], [0.09, 0.22],    # knees
    [-0.09, 0.44], [0.09, 0.44],    # ankles
], dtype=np.float32)


class DummyBackend:
    def __init__(self, period_frames: int = 80, seed: int = 0):
        self.period = period_frames
        self.n = 0
        self.rng = np.random.default_rng(seed)

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        k = self.n % self.period
        self.n += 1
        phase = k / self.period
        if 0.9 <= phase:                      # cuối chu kỳ: không có người
            return []
        scale = 0.6 * h
        cx = w * (0.25 + 0.5 * min(phase / 0.5, 1.0))
        cy = h * 0.55
        # 0.5..0.6: ngã (xoay 90 độ quanh hông, hạ thấp); sau đó nằm yên
        fall = min(max((phase - 0.5) / 0.1, 0.0), 1.0)
        ang = fall * math.pi / 2
        rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], np.float32)
        pts = _STAND @ rot.T
        cy += fall * 0.25 * h
        xy = np.empty_like(pts)
        xy[:, 0] = cx + pts[:, 0] * scale
        xy[:, 1] = cy + pts[:, 1] * scale
        xy += self.rng.normal(0, 1.5, xy.shape).astype(np.float32)   # rung nhẹ để thấy tác dụng của bộ lọc
        xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
        conf = np.full(17, 0.9, np.float32)
        conf[3:5] = 0.5                                              # tai thường kém tin cậy
        bbox = np.array([*xy.min(0), *xy.max(0)], np.float32)
        return [Detection(xy, conf, bbox, track_id=1, score=0.9)]

    def close(self):
        pass
