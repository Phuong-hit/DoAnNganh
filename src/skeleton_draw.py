"""
skeleton_draw.py
----------------
Vẽ skeleton + bounding box + confidence từ PoseResult (tọa độ 0-1)
lên khung hình bất kỳ kích thước.
"""
from __future__ import annotations

import cv2
import numpy as np

from .constants import SKELETON_EDGES
from .pose_estimate import PoseResult

BOX_COLOR = (255, 128, 0)


class SkeletonDraw:
    def __init__(
        self,
        min_conf=0.35,
        point_radius=4,
        line_thickness=2,
        show_status=True,
        show_bbox=True,
        show_score=True,
        show_kp_conf=False,
    ):
        self.min_conf = min_conf
        self.point_radius = point_radius
        self.line_thickness = line_thickness
        self.show_status = show_status
        self.show_bbox = show_bbox
        self.show_score = show_score
        self.show_kp_conf = show_kp_conf

    # ------------------------------------------------------------------
    def _draw_bbox(self, output, pose, w, h):
        if pose.bbox is None:
            return
        x1, y1, x2, y2 = pose.bbox
        p1 = (int(round(x1 * w)), int(round(y1 * h)))
        p2 = (int(round(x2 * w)), int(round(y2 * h)))
        cv2.rectangle(output, p1, p2, BOX_COLOR, 2)

        label = "person"
        if self.show_score and pose.score is not None:
            label += f" {pose.score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        ty = max(p1[1], th + 6)  # không cho nhãn tràn khỏi mép trên
        cv2.rectangle(output, (p1[0], ty - th - 6), (p1[0] + tw + 4, ty), BOX_COLOR, -1)
        cv2.putText(output, label, (p1[0] + 2, ty - 4), font, 0.6, (255, 255, 255), 2)

    # ------------------------------------------------------------------
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

        if self.show_bbox:
            self._draw_bbox(output, pose, w, h)

        j = np.asarray(pose.joints, dtype=np.float64)
        xy, conf = j[:, :2], j[:, 2]
        show = (
            np.isfinite(xy).all(axis=1)
            & np.isfinite(conf)
            & (conf >= self.min_conf)
        )
        px = np.zeros((len(j), 2), dtype=int)
        px[show, 0] = np.round(xy[show, 0] * w).astype(int)
        px[show, 1] = np.round(xy[show, 1] * h).astype(int)

        for a, b in SKELETON_EDGES:
            if show[a] and show[b]:
                cv2.line(
                    output, (int(px[a][0]), int(px[a][1])),
                    (int(px[b][0]), int(px[b][1])),
                    (0, 255, 0), self.line_thickness,
                )
        for i in np.flatnonzero(show):
            cv2.circle(output, (int(px[i][0]), int(px[i][1])), self.point_radius, (0, 0, 255), -1)
            if self.show_kp_conf:
                cv2.putText(
                    output, f"{conf[i]:.2f}", (int(px[i][0]) + 5, int(px[i][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
                )
        return output