"""SkeletonFilter cho MaixCAM — cùng thuật toán với src/skeleton_filter.py (PC), viết bằng số thực Python.

Vì sao không dùng numpy: mỗi lần gọi numpy trên CPU RISC-V của MaixCAM tốn ~50-100 µs chi phí cố định, mà bộ lọc
làm việc trên mảng rất nhỏ (17 khớp x 2 tọa độ) -> bản numpy cũ ~1200 lệnh gọi / khung ≈ 100 ms trên board.
Bản này chỉ đổi numpy <-> list 2 lần mỗi khung (đọc vào, trả ra). Kết quả khớp bản PC (tests/test_maixcam_fall.py).

Các bước (giống PC): bỏ khớp conf thấp -> bỏ khớp tạo xương dài bất thường -> trừ chuyển động chung (median) để tìm
khớp "nhảy" -> nếu đa số khớp cùng nhảy thì coi là chuyển động thật (ngã) và chấp nhận -> median theo thời gian
(history_size khung) -> One Euro filter -> khớp mất thì giữ vị trí cũ tối đa max_hold_frames khung.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .constants import BONES, NUM_KEYPOINTS
from .pose_estimate import PoseResult

_TWO_PI = 2.0 * math.pi
_NAN = float("nan")


def _median(vals):
    """Median như np.median (số phần tử chẵn -> trung bình 2 phần tử giữa)."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _alpha(cutoff, dt):
    cutoff = max(float(cutoff), 1e-6)
    dt = max(float(dt), 1e-6)
    tau = 1.0 / (_TWO_PI * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """One Euro filter cho một vector nhỏ (tuple số thực)."""

    def __init__(self, min_cutoff=1.5, beta=4.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x = None
        self.dx = None

    def reset(self):
        self.x = None
        self.dx = None

    def __call__(self, x, dt):
        x = tuple(float(v) for v in x)
        dt = max(float(dt), 1e-3)
        if self.x is None:
            self.x = x
            self.dx = (0.0,) * len(x)
            return self.x
        ad = _alpha(self.d_cutoff, dt)
        self.dx = tuple(ad * (xi - pi) / dt + (1.0 - ad) * di for xi, pi, di in zip(x, self.x, self.dx))
        speed = math.sqrt(sum(d * d for d in self.dx))
        a = _alpha(self.min_cutoff + self.beta * speed, dt)
        self.x = tuple(a * xi + (1.0 - a) * pi for xi, pi in zip(x, self.x))
        return self.x


class SkeletonFilter:

    def __init__(self, alpha=0.5, min_conf=0.35, max_hold_frames=5, history_size=3, jump_ratio=0.65,
                 min_jump=0.04, max_bone_ratio=2.5, collapse_ratio=0.5, max_gap_frames=15, min_cutoff=1.5,
                 beta=4.0, smooth_bbox=True, default_dt=0.1):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha phải nằm trong (0, 1].")
        if not 0.0 <= min_conf <= 1.0:
            raise ValueError("min_conf phải nằm trong [0, 1].")
        if max_hold_frames < 0:
            raise ValueError("max_hold_frames phải >= 0.")
        if history_size < 1:
            raise ValueError("history_size phải >= 1.")
        self.alpha = float(alpha)
        self.min_conf = float(min_conf)
        self.max_hold_frames = int(max_hold_frames)
        self.history_size = int(history_size)
        self.jump_ratio = float(jump_ratio)
        self.min_jump = float(min_jump)
        self.max_bone_ratio = float(max_bone_ratio)
        self.collapse_ratio = float(collapse_ratio)
        self.max_gap_frames = int(max_gap_frames)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.smooth_bbox = bool(smooth_bbox)
        self.default_dt = float(default_dt)
        self.reset()

    def reset(self):
        """Reset khi bắt đầu video mới hoặc mất người quá lâu."""
        self.prev_xy = [None] * NUM_KEYPOINTS                  # (x, y) đã lọc của khung trước, None = chưa có
        self.hist = [deque(maxlen=self.history_size) for _ in range(NUM_KEYPOINTS)]
        self.missing = [0] * NUM_KEYPOINTS
        self._gap = 0
        self._euro = [OneEuro(self.min_cutoff, self.beta) for _ in range(NUM_KEYPOINTS)]
        self._bbox_euro = OneEuro(self.min_cutoff, self.beta)
        self._last_t = None

    @staticmethod
    def _dist(p, q, aspect):
        return math.hypot((p[0] - q[0]) * aspect, p[1] - q[1])

    def _body_scale(self, xy, ok, aspect):
        lengths = []
        for a, b in BONES:
            if ok[a] and ok[b]:
                d = self._dist(xy[a], xy[b], aspect)
                if d > 0.01:
                    lengths.append(d)
        return _median(lengths) if lengths else None

    def _drop_long_bones(self, xy, conf, ok, aspect):
        """Loại khớp (conf thấp hơn) của xương dài bất thường. Không ép skeleton về tư thế đứng."""
        ok = list(ok)
        scale = self._body_scale(xy, ok, aspect)
        if scale is None:
            return ok
        max_length = self.max_bone_ratio * scale
        for a, b in BONES:
            if ok[a] and ok[b] and self._dist(xy[a], xy[b], aspect) > max_length:
                if conf[a] < conf[b]:
                    ok[a] = False
                else:
                    ok[b] = False
        return ok

    def smooth(self, pose):
        """Trả PoseResult mới, không sửa PoseResult đầu vào."""
        K = NUM_KEYPOINTS
        if not pose.valid:
            for i in range(K):
                self.missing[i] += 1
                if self.missing[i] > self.max_hold_frames:
                    self.hist[i].clear()
                    self.prev_xy[i] = None
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
        dt = max(float(dt), 1e-3)

        J = np.asarray(pose.joints, dtype=np.float32).tolist()      # 1 lần đổi numpy -> list
        xy = [(r[0], r[1]) for r in J]
        conf = [r[2] for r in J]
        aspect = float(pose.aspect)

        ok = [math.isfinite(x) and math.isfinite(y) and math.isfinite(c) and c >= self.min_conf
              for (x, y), c in zip(xy, conf)]
        ok = self._drop_long_bones(xy, conf, ok, aspect)
        scale = self._body_scale(xy, ok, aspect)
        jump_limit = max(self.min_jump, self.jump_ratio * scale if scale is not None else 0.0)

        prev = self.prev_xy
        has_prev = [p is not None for p in prev]
        cand = [ok[i] and has_prev[i] and self.missing[i] == 0 for i in range(K)]
        idx = [i for i in range(K) if cand[i]]

        if len(idx) >= 3:                                         # chuyển động chung của cơ thể (median)
            gx = _median([xy[i][0] - prev[i][0] for i in idx])
            gy = _median([xy[i][1] - prev[i][1] for i in idx])
        else:
            gx = gy = 0.0
        jumpers = [False] * K
        for i in idx:
            r = math.hypot((xy[i][0] - prev[i][0] - gx) * aspect, xy[i][1] - prev[i][1] - gy)
            jumpers[i] = r > jump_limit

        collapse = bool(idx) and (sum(jumpers) / float(len(idx))) >= self.collapse_ratio
        if collapse:                                              # cả người cùng chuyển động mạnh: chấp nhận
            accepted = list(ok)
            snap = list(jumpers)
        else:
            accepted = [ok[i] and not jumpers[i] for i in range(K)]
            snap = [False] * K
        for i in range(K):
            if accepted[i] and self.missing[i] > 0:               # khớp vừa tìm lại -> snap về vị trí mới
                snap[i] = True

        out = [[_NAN, _NAN, 0.0] for _ in range(K)]
        for i in range(K):
            if accepted[i]:
                if snap[i]:
                    self.hist[i].clear()
                self.hist[i].append(xy[i])
                h = self.hist[i]
                med = (_median([p[0] for p in h]), _median([p[1] for p in h]))
                if not (has_prev[i] and not snap[i]):
                    self._euro[i].reset()
                new = self._euro[i](med, dt)
                out[i] = [new[0], new[1], conf[i]]
                self.prev_xy[i] = (new[0], new[1])
                self.missing[i] = 0
            else:
                self.missing[i] += 1
                if has_prev[i] and self.missing[i] <= self.max_hold_frames:
                    out[i] = [prev[i][0], prev[i][1], 0.0]       # giữ vị trí cũ, conf 0 = không phải quan sát mới
                else:
                    self.hist[i].clear()
                    self.prev_xy[i] = None
                    self._euro[i].reset()

        bbox = pose.bbox
        if self.smooth_bbox and bbox is not None:
            bbox = tuple(float(v) for v in self._bbox_euro(bbox, dt))

        return PoseResult(frame_idx=pose.frame_idx, t=pose.t, valid=pose.valid,
                          joints=np.array(out, dtype=np.float32),   # 1 lần đổi list -> numpy
                          width=pose.width, height=pose.height, bbox=bbox,
                          track_id=pose.track_id, score=pose.score)
