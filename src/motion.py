"""
motion.py
---------
Đặc trưng TỐC ĐỘ của cửa sổ skeleton, để phân biệt NGÃ (nhanh, mất kiểm soát) với NẰM XUỐNG / leo lên
giường / ngả lưng (chậm, có kiểm soát) — kể cả ngã ngửa từ tư thế ngồi trên giường (hông gần như không hạ
nhưng thân xoay rất nhanh).

Tính trên clip ĐÃ chuẩn hóa bằng features.normalize_clip (một tâm, một tỉ lệ cho cả cửa sổ), nên đơn vị
độ dài là "độ dài xương trung vị" của chính người đó -> không phụ thuộc người xa/gần camera.

  shoulder_drop  tốc độ hạ xuống lớn nhất của trung điểm hai vai   (thân/giây; thân = độ dài xương trung vị)
  hip_drop       tốc độ hạ xuống lớn nhất của trung điểm hai hông  (thân/giây)
  rotation       tốc độ xoay lớn nhất của thân từ đứng sang nằm    (độ/giây)
  angle_end      góc thân cuối cửa sổ (0 = thẳng đứng, 90 = nằm ngang)

Tốc độ lấy trung bình trên `span` khung (mặc định 3 = 0.3 s @ 10 fps) để bớt nhiễu skeleton.
Khớp thiếu (mask = 0) không được dùng; không đo được thì giá trị = 0.

Giới hạn: camera 2D chỉ thấy chuyển động trong mặt phẳng ảnh. Cú ngã hướng thẳng về phía/ra xa camera có
tốc độ đo được thấp. Vì vậy tốc độ dùng làm CỔNG XÁC NHẬN tùy chọn (SpeedGate), không thay model.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

SHOULDERS, HIPS = [5, 6], [11, 12]
FEATURE_NAMES = ("shoulder_drop", "hip_drop", "rotation", "angle_end")


def _peak_rate(series: np.ndarray, span: int, fps: float) -> np.ndarray:
    """series [N,T] (NaN = thiếu) -> tốc độ tăng lớn nhất (đơn vị/giây) qua `span` khung, NaN -> 0."""
    if series.shape[1] <= span:
        return np.zeros(len(series), np.float32)
    rate = (series[:, span:] - series[:, :-span]) / span * fps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmax(rate, axis=1), nan=0.0).astype(np.float32)


def motion_features(clips, fps: float = 10.0, span: int = 3) -> dict:
    """clips [N,T,17,3] hoặc [T,17,3] đã chuẩn hóa -> dict tên -> mảng [N] (hoặc số nếu vào một clip)."""
    clips = np.asarray(clips, dtype=np.float32)
    single = clips.ndim == 3
    if single:
        clips = clips[None]
    m = clips[..., 2] > 0
    ok_sh, ok_hp = m[:, :, SHOULDERS].all(2), m[:, :, HIPS].all(2)
    sh = np.where(ok_sh[..., None], clips[:, :, SHOULDERS, :2].mean(2), np.nan)
    hp = np.where(ok_hp[..., None], clips[:, :, HIPS, :2].mean(2), np.nan)
    torso = sh - hp                                                  # vai - hông (y ảnh hướng xuống)
    angle = np.degrees(np.arctan2(np.abs(torso[..., 0]), -torso[..., 1]))   # 0 đứng, 90 nằm, >90 lộn ngược
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        angle_end = np.nan_to_num(np.nanmean(angle[:, -3:], axis=1), nan=0.0)
    out = dict(shoulder_drop=_peak_rate(sh[..., 1], span, fps), hip_drop=_peak_rate(hp[..., 1], span, fps),
               rotation=_peak_rate(angle, span, fps), angle_end=angle_end.astype(np.float32))
    if single:
        return {k: float(v[0]) for k, v in out.items()}
    return out


@dataclass
class SpeedGate:
    """Chỉ xác nhận NGÃ khi chuyển động đủ nhanh.

    NGÃ  <=>  P(ngã) >= ngưỡng  VÀ  (vai hạ >= shoulder_drop  HOẶC  thân xoay >= rotation
                                     HOẶC  P(ngã) >= keep_if_p_above)
    keep_if_p_above: model rất chắc chắn thì vẫn báo ngã dù đo tốc độ thấp (cú ngã hướng về camera...).
    """
    enabled: bool = False
    shoulder_drop: float = 2.5      # thân/giây
    rotation: float = 200.0         # độ/giây
    keep_if_p_above: float | None = 0.95

    @classmethod
    def from_dict(cls, d: dict | None) -> "SpeedGate":
        d = dict(d or {})
        known = {k: d[k] for k in ("enabled", "shoulder_drop", "rotation", "keep_if_p_above") if k in d}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)

    def passes(self, feats: dict, p_fall) -> np.ndarray | bool:
        """True = chuyển động đủ nhanh (hoặc cổng tắt). Nhận số hoặc mảng."""
        p = np.asarray(p_fall, dtype=np.float64)
        if not self.enabled:
            return np.ones(p.shape, bool) if p.ndim else True
        ok = (np.asarray(feats["shoulder_drop"]) >= self.shoulder_drop) | \
             (np.asarray(feats["rotation"]) >= self.rotation)
        if self.keep_if_p_above is not None:
            ok = ok | (p >= self.keep_if_p_above)
        return ok if np.ndim(ok) else bool(ok)

    def describe(self) -> str:
        if not self.enabled:
            return "tắt"
        s = f"vai hạ ≥ {self.shoulder_drop:g} thân/s"
        if self.rotation < 1e8:                       # >= 1e8 nghĩa là không dùng điều kiện xoay
            s += f" hoặc xoay ≥ {self.rotation:g}°/s"
        if self.keep_if_p_above is not None:
            s += f" (hoặc P ≥ {self.keep_if_p_above:g})"
        return s


def search_gate(y, p_fall, threshold: float, feats: dict, videos, max_video_recall_drop: float = 0.0,
                shoulder_grid=None, rotation_grid=None, keep_grid=(None, 0.9, 0.95, 0.98)) -> list[dict]:
    """Tìm ngưỡng cổng tốc độ trên một tập (nên là VAL). Trả danh sách ứng viên đã xếp hạng, tốt nhất trước.

    Ràng buộc: recall THEO VIDEO (video ngã có ít nhất một cửa sổ được báo) không giảm quá
    max_video_recall_drop so với khi không có cổng -> cổng không được làm mất cú ngã.
    Trong các ứng viên đạt, chọn F1 theo cửa sổ cao nhất, rồi ít video sinh hoạt bị báo nhầm nhất."""
    from .train_utils import metrics, video_level
    y = np.asarray(y)
    p = np.asarray(p_fall, dtype=np.float64)
    base_pred = (p >= threshold).astype(int)
    base = dict(metrics(y, base_pred), video=video_level(y, base_pred.astype(float), videos, 0.5))
    shoulder_grid = np.round(np.arange(0.5, 5.01, 0.25), 2) if shoulder_grid is None else shoulder_grid
    rotation_grid = [*range(80, 401, 20), None] if rotation_grid is None else rotation_grid
    out = []
    for keep in keep_grid:
        for sd in shoulder_grid:
            for rot in rotation_grid:
                g = SpeedGate(True, float(sd), float(rot) if rot is not None else float("inf"), keep)
                pred = (base_pred.astype(bool) & g.passes(feats, p)).astype(int)
                m, mv = metrics(y, pred), video_level(y, pred.astype(float), videos, 0.5)
                if mv["rec"] < base["video"]["rec"] - max_video_recall_drop - 1e-9:
                    continue
                out.append(dict(gate=dict(g.to_dict(), rotation=None if rot is None else float(rot)),
                                f1=m["f1"], rec=m["rec"], prec=m["prec"], spec=m["spec"],
                                video_rec=mv["rec"], video_fp=mv["fp"], video_fn=mv["fn"]))
    out.sort(key=lambda r: (round(r["f1"], 4), -r["video_fp"], r["rec"]), reverse=True)
    for r in out:                                       # rotation=None nghĩa là chỉ dùng tốc độ vai
        if r["gate"]["rotation"] is None:
            r["gate"]["rotation"] = 1e9
    return [dict(baseline=dict(f1=base["f1"], rec=base["rec"], prec=base["prec"], spec=base["spec"],
                               video_rec=base["video"]["rec"], video_fp=base["video"]["fp"],
                               video_fn=base["video"]["fn"]))] + out
