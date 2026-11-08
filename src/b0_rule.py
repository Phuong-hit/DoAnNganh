"""
b0_rule.py
----------
B0 cải tiến: mô hình LUẬT dựa trên chuỗi thời gian (rule-based).

Khắc phục 2 nhược điểm cốt lõi của bản cũ:
  1. "Cứ nằm là báo ngã":
     - Bản cũ tự động coi việc nằm từ đầu cửa sổ (ang_start > angle_th) là chuyển động ngã.
     - Bản mới yêu cầu chuyển động hạ thấp người NHANH (max_v_sy > v_fall_th hoặc
       max_v_hy kết hợp quay thân nhanh). Nằm xuống từ từ / nằm nghỉ không có vận tốc rơi cao.
  2. "Ngã bị khuất khỏi cam / mất dấu người":
     - Khi ngã sau bàn/ghế hoặc lọt mép dưới khung hình, số keypoints sụt giảm mạnh.
     - Phát hiện quy luật ngã khuất: ở đầu cửa sổ có người rõ ràng, có chuyển động rơi hoặc
       đổi góc thân, sau đó tỉ lệ quan sát khớp (mask) sụt giảm đột ngột (mask_drop > occ_drop_th).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12


@dataclass
class B0Params:
    angle_th: float = 55.0         # độ; thân nghiêng hơn mức này coi là "gần nằm"
    shoulder_low_th: float = 0.2   # vai cao hơn gốc -0.2 trở xuống coi là "thấp"
    v_fall_th: float = 0.20        # vận tốc hạ thấp vai (đơn vị xương/khung) coi là "rơi nhanh"
    angle_change_th: float = 35.0  # thân quay > mức này (độ)
    occ_drop_th: float = 0.40      # độ sụt giảm mask khi bị khuất/rơi khỏi cam
    end_frames: int = 4            # số khung cuối/đầu để lấy trung bình


def _mid(xy, mask, a, b):
    """Điểm giữa 2 khớp theo từng khung; chỉ có 1 khớp thì dùng khớp đó; không có -> NaN."""
    both = mask[:, a] & mask[:, b]
    one_a = mask[:, a] & ~mask[:, b]
    one_b = mask[:, b] & ~mask[:, a]
    p = np.full((len(xy), 2), np.nan)
    p[both] = (xy[both, a] + xy[both, b]) / 2
    p[one_a] = xy[one_a, a]
    p[one_b] = xy[one_b, b]
    return p


def _fill(v):
    ok = ~np.isnan(v)
    if ok.sum() < 2:
        return None
    t = np.arange(len(v))
    return np.interp(t, t[ok], v[ok])


def clip_features(clip, p: B0Params | None = None):
    """clip [W,17,3] -> dict đặc trưng."""
    p = p or B0Params()
    clip = np.asarray(clip, dtype=np.float64)
    xy, mask = clip[..., :2], clip[..., 2] > 0
    e = p.end_frames

    m_ratio = mask.mean(axis=1)
    mask_start = float(m_ratio[:e].mean()) if len(m_ratio) >= e else 0.0
    mask_end = float(m_ratio[-e:].mean()) if len(m_ratio) >= e else 0.0
    mask_drop = mask_start - mask_end

    s, h = _mid(xy, mask, L_SHO, R_SHO), _mid(xy, mask, L_HIP, R_HIP)
    sx, sy, hx, hy = (_fill(v) for v in (s[:, 0], s[:, 1], h[:, 0], h[:, 1]))

    # Trường hợp mất hoàn toàn khớp ở cuối do ngã khuất khỏi cam
    if sx is None or hx is None:
        occ = bool(mask_start >= 0.4 and mask_drop >= 0.3)
        return dict(
            ang_start=np.nan, ang_end=np.nan, ang_max=np.nan,
            sy_end=np.nan, hy_range=np.nan,
            max_v_sy=0.2 if occ else 0.0,
            max_v_hy=0.15 if occ else 0.0,
            mask_start=mask_start, mask_end=mask_end, mask_drop=mask_drop,
            occluded_fall=occ, is_valid=occ
        )

    ang = np.degrees(np.arctan2(np.abs(sx - hx), -(sy - hy)))
    # Vận tốc hạ thấp theo trục y (y hướng xuống nên y tăng = đang rơi xuống)
    # Lấy hiệu qua 2 khung để giảm nhiễu rung giật
    v_sy = (sy[2:] - sy[:-2]) / 2.0 if len(sy) > 2 else np.zeros(1)
    v_hy = (hy[2:] - hy[:-2]) / 2.0 if len(hy) > 2 else np.zeros(1)
    max_v_sy = float(np.max(v_sy)) if len(v_sy) else 0.0
    max_v_hy = float(np.max(v_hy)) if len(v_hy) else 0.0

    # Ngã khuất: đầu cửa sổ thấy người rõ, có chuyển động rơi/đổ thân, sau đó khớp sụt giảm mạnh
    occ = bool(
        mask_start >= 0.4 and mask_drop >= p.occ_drop_th and
        (max_v_sy >= 0.08 or (hy.max() - hy.min()) >= 0.5 or (ang.max() - ang[:e].mean()) >= 15.0)
    )

    return dict(
        ang_start=float(ang[:e].mean()),
        ang_end=float(ang[-e:].mean()),
        ang_max=float(ang.max()),
        sy_end=float(sy[-e:].mean()),
        hy_range=float(hy.max() - hy.min()),
        max_v_sy=max_v_sy,
        max_v_hy=max_v_hy,
        mask_start=mask_start,
        mask_end=mask_end,
        mask_drop=mask_drop,
        occluded_fall=occ,
        is_valid=True
    )


FEATURE_KEYS = (
    "ang_start", "ang_end", "ang_max", "sy_end", "hy_range",
    "max_v_sy", "max_v_hy", "mask_start", "mask_end", "mask_drop",
    "occluded_fall", "is_valid"
)


def extract_all(X, p: B0Params | None = None):
    """X [N,W,17,3] -> dict {tên đặc trưng: mảng [N]}."""
    feats = [clip_features(c, p) for c in X]
    return {k: np.array([f[k] for f in feats]) for k in FEATURE_KEYS}


def predict_from_features(F, p: B0Params | None = None):
    p = p or B0Params()
    # 1. Tư thế sau ngã: thân nghiêng ngang HOẶC vai hạ thấp sát sàn
    posture = (F["ang_end"] > p.angle_th) | (F["sy_end"] > -p.shoulder_low_th)

    # 2. Chuyển động ngã thật: bắt buộc phải có gia tốc / vận tốc rơi nhanh
    # Không tính người đã nằm sẵn từ trước (ang_start > angle_th) trừ khi có vận tốc rơi lớn!
    rapid = (F["max_v_sy"] > p.v_fall_th) | (
        (F["max_v_hy"] > p.v_fall_th * 0.7) & ((F["ang_max"] - F["ang_start"]) > p.angle_change_th)
    )
    std_fall = posture & rapid

    # 3. Ngã khuất khỏi góc nhìn của camera (occluded fall)
    occ_fall = F["occluded_fall"]

    pred = std_fall | occ_fall
    return np.where(~F["is_valid"] & ~occ_fall, False, pred).astype(np.int64)


class B0Rule:
    """Giao diện mô hình luật: predict(X) -> nhãn 0/1; score(X) -> điểm số ngã [0, 1]."""

    def __init__(self, params: B0Params | None = None):
        self.params = params or B0Params()

    def predict(self, X):
        return predict_from_features(extract_all(X, self.params), self.params)

    def score(self, X):
        """Điểm tin cậy ngã [0, 1] tổng hợp từ góc thân, vận tốc rơi và cờ khuất."""
        F = extract_all(X, self.params)
        ang_score = np.clip(np.nan_to_num(F["ang_end"]) / 90.0, 0, 1)
        v_score = np.clip(np.nan_to_num(F["max_v_sy"]) / (self.params.v_fall_th * 1.5), 0, 1)
        occ_bonus = np.where(F["occluded_fall"], 0.8, 0.0)
        return np.clip(0.5 * ang_score + 0.5 * v_score + occ_bonus, 0, 1)

    def get_params(self):
        return asdict(self.params)

    @classmethod
    def from_params(cls, d):
        # Lọc chỉ những tham số thuộc B0Params để tương thích ngược
        valid_keys = set(B0Params.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(B0Params(**filtered))


class OnlineFallDetector:
    """Bộ phát hiện ngã trực tuyến (online / real-time) chạy trên MaixCAM hoặc Webcam.

    Duy trì hàng đợi vòng (ring buffer) W khung hình, chuẩn hóa động và cảnh báo ngã
    kèm chốt trạng thái (latch) khi ngã bị khuất/mất dấu người.
    """

    def __init__(
        self,
        window_size: int = 20,
        fps: float = 10.0,
        params: B0Params | None = None,
        latch_seconds: float = 5.0,
    ):
        self.window_size = window_size
        self.fps = fps
        self.params = params or B0Params()
        self.latch_frames = int(latch_seconds * fps)

        self.buffer = deque(maxlen=window_size)
        self.mask_buffer = deque(maxlen=window_size)
        self._latch_counter = 0

    def reset(self):
        self.buffer.clear()
        self.mask_buffer.clear()
        self._latch_counter = 0

    def update(self, joints_17x3: np.ndarray | None, aspect: float = 1.0) -> dict:
        """Nhận kết quả pose [17, 3] của khung hình hiện tại (x, y [0-1], conf).

        Trả về: dict(is_fall=bool, is_occluded=bool, score=float, latched=bool)
        """
        if joints_17x3 is None or np.isnan(joints_17x3[:, :2]).all():
            curr_xy = np.zeros((17, 2), dtype=np.float32)
            curr_m = np.zeros(17, dtype=bool)
        else:
            curr_xy = joints_17x3[:, :2].copy()
            curr_m = np.isfinite(curr_xy).all(axis=-1) & (joints_17x3[:, 2] > 0.2)

        self.buffer.append(curr_xy)
        self.mask_buffer.append(curr_m)

        if len(self.buffer) < self.window_size:
            return dict(is_fall=False, is_occluded=False, score=0.0, latched=False)

        # Chuẩn hóa clip tương tự features.py
        from .features import normalize_clip
        seq = np.stack(self.buffer, axis=0)
        mask = np.stack(self.mask_buffer, axis=0)
        norm_clip = normalize_clip(seq, mask, aspect=aspect)

        feat = clip_features(norm_clip, self.params)
        if feat is None:
            if self._latch_counter > 0:
                self._latch_counter -= 1
                return dict(is_fall=True, is_occluded=True, score=0.8, latched=True)
            return dict(is_fall=False, is_occluded=False, score=0.0, latched=False)

        F_dict = {k: np.array([feat[k]]) for k in FEATURE_KEYS}
        is_fall = bool(predict_from_features(F_dict, self.params)[0] == 1)
        is_occluded = bool(feat["occluded_fall"])

        if is_fall:
            self._latch_counter = self.latch_frames
            latched = False
        elif self._latch_counter > 0:
            self._latch_counter -= 1
            is_fall = True
            latched = True
        else:
            latched = False

        return dict(
            is_fall=is_fall,
            is_occluded=is_occluded,
            score=float(feat["max_v_sy"]),
            latched=latched,
        )
