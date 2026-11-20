

from __future__ import annotations

from collections import deque

import numpy as np

from .constants import BONES, NUM_KEYPOINTS
from .pose_estimate import PoseResult




class OneEuro:
    

    def __init__(
        self,
        min_cutoff=1.5,
        beta=4.0,
        d_cutoff=1.0
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x = None
        self.dx = None

    def reset(self):
        self.x = None
        self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        cutoff = max(float(cutoff), 1e-6)
        dt = max(float(dt), 1e-6)

        tau = 1.0 / (2.0 * np.pi * cutoff)

        return float(
            1.0 / (1.0 + tau / dt)
        )

    def __call__(self, x, dt):

        x = np.asarray(
            x,
            dtype=np.float32
        )

        dt = max(float(dt), 1e-3)

        # Frame đầu tiên
        if self.x is None:

            self.x = x.copy()

            self.dx = np.zeros_like(
                x,
                dtype=np.float32
            )

            return self.x.copy()

        # Tốc độ chuyển động
        dx = (
            x - self.x
        ) / dt

        # Lọc đạo hàm
        alpha_d = self._alpha(
            self.d_cutoff,
            dt
        )

        self.dx = (
            alpha_d * dx
            +
            (1.0 - alpha_d) * self.dx
        ).astype(np.float32)

        # Cutoff phụ thuộc tốc độ
        speed = float(
            np.linalg.norm(self.dx)
        )

        cutoff = (
            self.min_cutoff
            +
            self.beta * speed
        )

        # Alpha cho vị trí
        alpha = self._alpha(
            cutoff,
            dt
        )

        # Lọc vị trí
        self.x = (
            alpha * x
            +
            (1.0 - alpha) * self.x
        ).astype(np.float32)

        return self.x.copy()



class SkeletonFilter:

    def __init__(
        self,
        alpha=0.5,
        min_conf=0.35,
        max_hold_frames=5,
        history_size=3,
        jump_ratio=0.65,
        min_jump=0.04,
        max_bone_ratio=2.5,
        collapse_ratio=0.5,
        max_gap_frames=15,
        min_cutoff=1.5,
        beta=4.0,
        smooth_bbox=True,
        default_dt=0.1
    ):

        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                "alpha phải nằm trong (0, 1]."
            )

        if not 0.0 <= min_conf <= 1.0:
            raise ValueError(
                "min_conf phải nằm trong [0, 1]."
            )

        if max_hold_frames < 0:
            raise ValueError(
                "max_hold_frames phải >= 0."
            )

        if history_size < 1:
            raise ValueError(
                "history_size phải >= 1."
            )

        
        self.alpha = float(alpha)

        self.min_conf = float(min_conf)

        self.max_hold_frames = int(
            max_hold_frames
        )

        self.history_size = int(
            history_size
        )

        self.jump_ratio = float(
            jump_ratio
        )

        self.min_jump = float(
            min_jump
        )

        self.max_bone_ratio = float(
            max_bone_ratio
        )

        self.collapse_ratio = float(
            collapse_ratio
        )

        self.max_gap_frames = int(
            max_gap_frames
        )

        self.min_cutoff = float(
            min_cutoff
        )

        self.beta = float(
            beta
        )

        self.smooth_bbox = bool(
            smooth_bbox
        )

        self.default_dt = float(
            default_dt
        )

        self.reset()

    

    def reset(self):
        """
        Reset khi bắt đầu video mới
        hoặc mất người quá lâu.
        """

        # Vị trí khớp trước đó
        self.prev_xy = np.full(
            (NUM_KEYPOINTS, 2),
            np.nan,
            dtype=np.float32
        )

        # Lịch sử từng khớp
        self.hist = [
            deque(maxlen=self.history_size)
            for _ in range(NUM_KEYPOINTS)
        ]

        # Số frame liên tiếp khớp bị mất
        self.missing = np.zeros(
            NUM_KEYPOINTS,
            dtype=np.int32
        )

        # Số frame liên tiếp mất người
        self._gap = 0

        # One Euro Filter cho từng khớp
        self._euro = [
            OneEuro(
                min_cutoff=self.min_cutoff,
                beta=self.beta
            )
            for _ in range(NUM_KEYPOINTS)
        ]

        
        self._bbox_euro = OneEuro(
            min_cutoff=self.min_cutoff,
            beta=self.beta
        )

        self._last_t = None

    

    @staticmethod
    def _dist(p, q, aspect):
        """
        Khoảng cách Euclidean có hiệu chỉnh aspect ratio.

        x được nhân aspect để tránh sai lệch
        do khung hình không vuông.
        """

        p = np.asarray(
            p,
            dtype=np.float32
        )

        q = np.asarray(
            q,
            dtype=np.float32
        )

        d = p - q

        return np.hypot(
            d[..., 0] * aspect,
            d[..., 1]
        )

    

    def _body_scale(self, xy, ok, aspect):

        lengths = []

        for a, b in BONES:

            if ok[a] and ok[b]:

                length = float(
                    self._dist(
                        xy[a],
                        xy[b],
                        aspect
                    )
                )

                if length > 0.01:
                    lengths.append(length)

        if not lengths:
            return None

        return float(
            np.median(
                np.asarray(
                    lengths,
                    dtype=np.float32
                )
            )
        )

    

    def _drop_long_bones(
        self,
        xy,
        conf,
        ok,
        aspect
    ):
        """
        Loại khớp gây ra xương có chiều dài bất thường.

        Không ép skeleton về tư thế đứng.

        Chỉ loại khớp có confidence thấp hơn
        trong cặp xương bất thường.
        """

        ok = ok.copy()

        scale = self._body_scale(
            xy,
            ok,
            aspect
        )

        if scale is None:
            return ok

        max_length = (
            self.max_bone_ratio
            * scale
        )

        for a, b in BONES:

            if ok[a] and ok[b]:

                length = float(
                    self._dist(
                        xy[a],
                        xy[b],
                        aspect
                    )
                )

                if length > max_length:

                    # Loại khớp có confidence thấp
                    if conf[a] < conf[b]:
                        ok[a] = False
                    else:
                        ok[b] = False

        return ok

    

    @staticmethod
    def _global_motion(
        xy,
        prev_xy,
        cand
    ):
        """
        Ước lượng chuyển động chung của cơ thể.

        Sử dụng median để hạn chế ảnh hưởng
        của các khớp nhiễu.
        """

        if int(cand.sum()) < 3:

            return np.zeros(
                2,
                dtype=np.float32
            )

        motion = np.median(
            xy[cand] - prev_xy[cand],
            axis=0
        )

        return np.asarray(
            motion,
            dtype=np.float32
        )

    

    def smooth(self, pose):
        """
        Trả về PoseResult mới.

        Không sửa PoseResult đầu vào.
        """

        

        if not pose.valid:

            self.missing += 1

            
            expired = (
                self.missing
                >
                self.max_hold_frames
            )

            for i in np.flatnonzero(expired):

                self.hist[i].clear()

                self.prev_xy[i] = np.nan

                self._euro[i].reset()

            self._gap += 1

            
            if self._gap > self.max_gap_frames:

                self.reset()

            
            return pose

        

        self._gap = 0

        

        dt = self.default_dt

        if (
            self._last_t is not None
            and pose.t > self._last_t
        ):

            dt = pose.t - self._last_t

        self._last_t = pose.t

        dt = max(
            float(dt),
            1e-3
        )

        

        xy = np.asarray(
            pose.joints[:, :2],
            dtype=np.float32
        )

        conf = np.asarray(
            pose.joints[:, 2],
            dtype=np.float32
        )

        aspect = float(
            pose.aspect
        )

        

        finite = (
            np.isfinite(xy).all(axis=1)
            &
            np.isfinite(conf)
        )

        ok = (
            finite
            &
            (conf >= self.min_conf)
        )

        
        ok = self._drop_long_bones(
            xy,
            conf,
            ok,
            aspect
        )

        

        scale = self._body_scale(
            xy,
            ok,
            aspect
        )

        jump_limit = max(
            self.min_jump,
            (
                self.jump_ratio * scale
                if scale is not None
                else 0.0
            )
        )

        

        has_prev = np.isfinite(
            self.prev_xy
        ).all(axis=1)

       
        cand = (
            ok
            &
            has_prev
            &
            (self.missing == 0)
        )

        

        g = self._global_motion(
            xy,
            self.prev_xy,
            cand
        )

        

        residual = np.zeros(
            NUM_KEYPOINTS,
            dtype=np.float32
        )

        if cand.any():

            d = (
                xy[cand]
                -
                self.prev_xy[cand]
                -
                g
            )

            residual[cand] = np.hypot(
                d[:, 0] * aspect,
                d[:, 1]
            )

        jumpers = (
            cand
            &
            (residual > jump_limit)
        )

        

        collapse = False

        if cand.sum() > 0:

            jump_ratio = (
                float(jumpers.sum())
                /
                float(cand.sum())
            )

            collapse = (
                jump_ratio
                >=
                self.collapse_ratio
            )

        if collapse:

            # Chuyển động mạnh của toàn cơ thể
            # được chấp nhận.
            accepted = ok.copy()

            snap = jumpers.copy()

        else:

            # Loại các khớp nhảy bất thường
            accepted = (
                ok
                &
                ~jumpers
            )

            snap = np.zeros(
                NUM_KEYPOINTS,
                dtype=bool
            )

        # Khớp vừa tìm lại sau khi bị mất
        # phải snap về vị trí mới.
        snap |= (
            accepted
            &
            (self.missing > 0)
        )

       

        out = np.full(
            (NUM_KEYPOINTS, 3),
            np.nan,
            dtype=np.float32
        )

        out[:, 2] = 0.0

        
        for i in range(NUM_KEYPOINTS):

            

            if accepted[i]:

                # Khớp nhảy hoặc vừa được tìm lại
                if snap[i]:

                    self.hist[i].clear()

                # Thêm tọa độ vào history
                self.hist[i].append(
                    xy[i].copy()
                )

                # Median temporal
                history = np.stack(
                    list(self.hist[i]),
                    axis=0
                )

                med = np.median(
                    history,
                    axis=0
                ).astype(
                    np.float32
                )

               

                if (
                    has_prev[i]
                    and
                    not snap[i]
                ):

                    new = self._euro[i](
                        med,
                        dt
                    )

                else:

                    # Reset khi snap
                    self._euro[i].reset()

                    new = self._euro[i](
                        med,
                        dt
                    )

                new = np.asarray(
                    new,
                    dtype=np.float32
                )

                

                out[i, 0] = new[0]

                out[i, 1] = new[1]

                out[i, 2] = conf[i]

                # Lưu vị trí mới
                self.prev_xy[i] = new

                # Khớp đã được tìm thấy
                self.missing[i] = 0

            
            else:

                self.missing[i] += 1

                # Giữ tọa độ cũ trong thời gian ngắn
                if (
                    has_prev[i]
                    and
                    self.missing[i]
                    <= self.max_hold_frames
                ):

                    out[i, 0] = (
                        self.prev_xy[i, 0]
                    )

                    out[i, 1] = (
                        self.prev_xy[i, 1]
                    )

                    # Confidence 0:
                    # Không phải quan sát mới
                    out[i, 2] = 0.0

                else:

                    # Mất quá thời gian cho phép
                    self.hist[i].clear()

                    self.prev_xy[i] = np.nan

                    self._euro[i].reset()

        
        bbox = pose.bbox

        if (
            self.smooth_bbox
            and
            bbox is not None
        ):

            bbox_array = np.asarray(
                bbox,
                dtype=np.float32
            )

            bbox_smoothed = self._bbox_euro(
                bbox_array,
                dt
            )

            bbox = tuple(
                float(v)
                for v in bbox_smoothed
            )

        

        return PoseResult(
            frame_idx=pose.frame_idx,
            t=pose.t,
            valid=pose.valid,
            joints=out,
            width=pose.width,
            height=pose.height,
            bbox=bbox,
            track_id=pose.track_id,
            score=pose.score
        )