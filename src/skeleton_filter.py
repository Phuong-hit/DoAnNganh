"""
skeleton_filter.py
------------------
Làm mượt skeleton theo thời gian, làm việc trên PoseResult (tọa độ 0-1).

So với bản cũ, đã sửa 3 lỗi:
  1. Mất hàng loạt khớp sau cú ngã: khi >= collapse_ratio số khớp cùng "nhảy"
     trong một khung (ngã, xoay người nhanh) thì đó là chuyển động thật,
     chấp nhận tất cả thay vì coi là nhiễu.
  2. Kéo khung về vị trí cũ: khớp vừa tìm lại sau khi bị giữ/mất sẽ "snap"
     về quan sát mới (xóa lịch sử, không trộn EMA với vị trí cũ); EMA cũng
     tự tăng alpha khi chuyển động lớn.
  3. Phụ thuộc độ phân giải: ngưỡng tính theo đơn vị chiều cao khung (0-1),
     khoảng cách đã hiệu chỉnh theo tỉ lệ khung hình (aspect).
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from .constants import BONES, NUM_KEYPOINTS
from .pose_estimate import PoseResult


class OneEuro:
    """One Euro Filter cho vector (Casiez 2012): mượt khi đứng yên, bám sát khi
    chuyển động nhanh. Đơn vị tọa độ 0-1, thời gian giây."""

    def __init__(self, min_cutoff=1.5, beta=4.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None

    def reset(self):
        self.x = None
        self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        x = np.asarray(x, dtype=np.float64)
        if self.x is None:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return x
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


class SkeletonFilter:
    def __init__(
        self,
        alpha: float = 0.5,
        min_conf: float = 0.35,
        max_hold_frames: int = 5,
        history_size: int = 3,
        jump_ratio: float = 0.65,
        min_jump: float = 0.04,       # đơn vị: chiều cao khung (0-1)
        max_bone_ratio: float = 2.5,
        collapse_ratio: float = 0.5,
        max_gap_frames: int = 15,     # số khung valid=False liên tiếp trước khi reset
        min_cutoff: float = 1.5,      # One Euro: nhỏ hơn = mượt hơn khi đứng yên
        beta: float = 4.0,            # One Euro: lớn hơn = bám nhanh hơn khi chuyển động
        smooth_bbox: bool = True,
        default_dt: float = 0.1,      # dùng khi t không tăng
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha phải nằm trong (0, 1].")
        if not 0.0 <= min_conf <= 1.0:
            raise ValueError("min_conf phải nằm trong [0, 1].")
        if max_hold_frames < 0 or history_size < 1:
            raise ValueError("max_hold_frames >= 0; history_size >= 1.")

        self.alpha = alpha
        self.min_conf = min_conf
        self.max_hold_frames = max_hold_frames
        self.history_size = history_size
        self.jump_ratio = jump_ratio
        self.min_jump = min_jump
        self.max_bone_ratio = max_bone_ratio
        self.collapse_ratio = collapse_ratio
        self.max_gap_frames = max_gap_frames
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.smooth_bbox = smooth_bbox
        self.default_dt = default_dt
        self.reset()

    def reset(self):
        """Gọi khi bắt đầu video mới."""
        self.prev_xy = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float64)
        self.hist = [deque(maxlen=self.history_size) for _ in range(NUM_KEYPOINTS)]
        self.missing = np.zeros(NUM_KEYPOINTS, dtype=int)
        self._gap = 0
        self._euro = [OneEuro(self.min_cutoff, self.beta) for _ in range(NUM_KEYPOINTS)]
        self._bbox_euro = OneEuro(self.min_cutoff, self.beta)
        self._last_t = None

    # ------------------------------------------------------------------
    @staticmethod
    def _dist(p, q, aspect):
        d = np.asarray(p) - np.asarray(q)
        return np.hypot(d[..., 0] * aspect, d[..., 1])

    def _body_scale(self, xy, ok, aspect):
        lengths = [
            float(self._dist(xy[a], xy[b], aspect))
            for a, b in BONES
            if ok[a] and ok[b]
        ]
        lengths = [l for l in lengths if l > 0.01]
        return float(np.median(lengths)) if lengths else None

    def _drop_long_bones(self, xy, conf, ok, aspect):
        """Chỉ loại khớp làm xương dài bất thường; không ép tư thế đứng."""
        ok = ok.copy()
        scale = self._body_scale(xy, ok, aspect)
        if scale is None:
            return ok
        for a, b in BONES:
            if ok[a] and ok[b]:
                if self._dist(xy[a], xy[b], aspect) > self.max_bone_ratio * scale:
                    ok[a if conf[a] < conf[b] else b] = False
        return ok

    @staticmethod
    def _global_motion(xy, prev_xy, cand):
        if cand.sum() < 3:
            return np.zeros(2)
        return np.median(xy[cand] - prev_xy[cand], axis=0)

    # ------------------------------------------------------------------
    def smooth(self, pose: PoseResult) -> PoseResult:
        """Trả PoseResult mới, không sửa đầu vào."""
        if not pose.valid:
            # Khung mất người cũng tính là mất khớp: khớp giữ tạm không sống
            # lâu hơn max_hold_frames, và khi tìm lại sẽ được snap.
            self.missing += 1
            for i in np.flatnonzero(self.missing > self.max_hold_frames):
                self.hist[i].clear()
                self.prev_xy[i] = np.nan
                self._euro[i].reset()
            self._gap += 1
            if self._gap > self.max_gap_frames:
                self.reset()
            return pose
        self._gap = 0

        dt = self.default_dt
        if self._last_t is not None and pose.t > self._last_t:
            dt = pose.t - self._last_t
        self._last_t = pose.t
        dt = max(dt, 1e-3)  # tránh chia cho dt ~ 0 (webcam)

        n = NUM_KEYPOINTS
        xy = pose.joints[:, :2].astype(np.float64)
        conf = pose.joints[:, 2].astype(np.float64)
        aspect = pose.aspect

        finite = np.isfinite(xy).all(axis=1) & np.isfinite(conf)
        ok = finite & (conf >= self.min_conf)
        ok = self._drop_long_bones(xy, conf, ok, aspect)

        scale = self._body_scale(xy, ok, aspect)
        jump_limit = max(self.min_jump, self.jump_ratio * scale if scale else 0.0)

        has_prev = np.isfinite(self.prev_xy).all(axis=1)
        # Khớp vừa bị mất/giữ (missing > 0) không bị kiểm tra nhảy: đó là tìm lại.
        cand = ok & has_prev & (self.missing == 0)
        g = self._global_motion(xy, self.prev_xy, cand)

        residual = np.zeros(n)
        if cand.any():
            d = xy[cand] - self.prev_xy[cand] - g
            residual[cand] = np.hypot(d[:, 0] * aspect, d[:, 1])
        jumpers = cand & (residual > jump_limit)

        if cand.sum() > 0 and jumpers.sum() / cand.sum() >= self.collapse_ratio:
            # Chuyển động thật của cả cơ thể (ngã...): chấp nhận hết, snap.
            accepted = ok.copy()
            snap = jumpers.copy()
        else:
            accepted = ok & ~jumpers
            snap = np.zeros(n, dtype=bool)
        snap |= accepted & (self.missing > 0)  # tìm lại sau khi mất

        out = np.full((n, 3), np.nan, dtype=np.float32)
        out[:, 2] = 0.0

        for i in range(n):
            if accepted[i]:
                if snap[i]:
                    self.hist[i].clear()
                self.hist[i].append(xy[i].copy())
                med = np.median(np.stack(self.hist[i]), axis=0)

                if has_prev[i] and not snap[i]:
                    new = self._euro[i](med, dt)
                else:
                    self._euro[i].reset()
                    new = self._euro[i](med, dt)

                out[i] = (new[0], new[1], conf[i])
                self.prev_xy[i] = new
                self.missing[i] = 0
            else:
                self.missing[i] += 1
                if has_prev[i] and self.missing[i] <= self.max_hold_frames:
                    # Giữ tạm vị trí, conf = 0: báo "không phải quan sát mới".
                    out[i] = (self.prev_xy[i][0], self.prev_xy[i][1], 0.0)
                else:
                    self.hist[i].clear()
                    self.prev_xy[i] = np.nan
                    self._euro[i].reset()

        bbox = pose.bbox
        if self.smooth_bbox and bbox is not None:
            bbox = tuple(float(v) for v in self._bbox_euro(np.asarray(bbox), dt))

        return replace(pose, joints=out, bbox=bbox)