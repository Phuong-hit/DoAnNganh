"""
skeleton_draw.py
----------------
Vẽ skeleton từ PoseResult (tọa độ 0-1) lên khung hình bất kỳ kích thước.
"""
from __future__ import annotations

import cv2
import numpy as np

from .constants import SKELETON_EDGES
from .pose_estimate import PoseResult


class SkeletonDraw:
    def __init__(
        self,
        min_conf=0.3,
        point_radius=4,
        line_thickness=2,
        show_status=True,
    ):
        self.min_conf = min_conf
        self.point_radius = point_radius
        self.line_thickness = line_thickness
        self.show_status = show_status

    def draw(self, frame, pose: PoseResult | None):
        output = frame.copy()
        h, w = output.shape[:2]

        if pose is None or not pose.valid:
            if self.show_status:
                cv2.putText(
                    output, "NO PERSON", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )
            return output

        j = np.asarray(pose.joints, dtype=np.float64)
        xy, conf = j[:, :2], j[:, 2]
        # conf NaN/None coi như 0; tọa độ NaN coi như thiếu.
        show = (
            np.isfinite(xy).all(axis=1)
            & np.isfinite(conf)
            & (conf >= self.min_conf)
        )
        px = np.zeros((len(j), 2), dtype=int)
        px[show, 0] = np.round(xy[show, 0] * w).astype(int)  # 0-1 -> pixel
        px[show, 1] = np.round(xy[show, 1] * h).astype(int)

        for a, b in SKELETON_EDGES:
            if show[a] and show[b]:
                cv2.line(
                    output, tuple(px[a]), tuple(px[b]),
                    (0, 255, 0), self.line_thickness,
                )
        for i in np.flatnonzero(show):
            cv2.circle(
                output, tuple(px[i]), self.point_radius, (0, 0, 255), -1
            )
        return output