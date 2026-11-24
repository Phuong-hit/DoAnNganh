"""
fall_postprocess.py
-------------------
Thuật toán hậu xử lý xác suất ngã của nhánh `final` (main.py: FallPostProcessor), tách ra để web / monitor /
chạy offline dùng chung với FallClassifier (src/classifier.py, mục `classifier.postprocess` trong configs/web.yaml).

  1. Kích hoạt NGÃ khi P(ngã) >= threshold liên tiếp consec_frames lần.
  2. Giữ cảnh báo chừng nào người chưa đứng thẳng hoàn toàn (cả thân trên lẫn chân).
  3. Xả bộ đếm nhanh (-3 mỗi khung) khi đã đứng thẳng dậy.

Logic giữ nguyên bản final; chỉ thêm update_arrays() nhận thẳng seq/mask của khung hiện tại (đã có sẵn khi
chạy offline trên NPZ), update() nhận PoseResult như bản gốc.
"""
from __future__ import annotations

import numpy as np

from .skeleton_io import from_poses


def is_standing_upright(seq, mask, aspect: float = 1.0) -> bool:
    """seq [17, >=2] (x, y 0-1), mask [17] của MỘT khung -> cả người (thân trên + chân) đã đứng thẳng chưa.
    Loại trừ trường hợp mới ngồi dậy trên sàn (vai thẳng nhưng chân gập/duỗi ngang)."""
    valid_shoulders = [i for i in (5, 6) if mask[i]]
    valid_hips = [i for i in (11, 12) if mask[i]]
    valid_knees = [i for i in (13, 14) if mask[i]]
    valid_ankles = [i for i in (15, 16) if mask[i]]

    # Phải nhìn thấy đủ cả vai, hông và phần chân (cổ chân hoặc đầu gối)
    if not valid_shoulders or not valid_hips or (not valid_ankles and not valid_knees):
        return False

    shoulder_x = float(np.mean([seq[i, 0] for i in valid_shoulders])) * aspect
    shoulder_y = float(np.mean([seq[i, 1] for i in valid_shoulders]))
    hip_x = float(np.mean([seq[i, 0] for i in valid_hips])) * aspect
    hip_y = float(np.mean([seq[i, 1] for i in valid_hips]))

    # 1. Thân trên (vai -> hông) phải dựng thẳng đứng
    torso_dx = abs(shoulder_x - hip_x)
    torso_dy = hip_y - shoulder_y
    if torso_dy <= 0 or torso_dy <= torso_dx * 2.0:
        return False

    # 2. Phần chân (hông -> cổ chân, khuất cổ chân thì dùng gối) phải duỗi thẳng xuống dưới
    lower = valid_ankles if valid_ankles else valid_knees
    lower_x = float(np.mean([seq[i, 0] for i in lower])) * aspect
    lower_y = float(np.mean([seq[i, 1] for i in lower]))
    leg_dx = abs(hip_x - lower_x)
    leg_dy = lower_y - hip_y
    if leg_dy <= 0 or leg_dy <= leg_dx * 2.0:
        return False

    # 3. Chân đủ dài so với thân trên (cổ chân: >= 0.85 lần, gối: >= 0.6 lần)
    min_leg_ratio = 0.85 if valid_ankles else 0.60
    return leg_dy >= torso_dy * min_leg_ratio


class FallPostProcessor:
    def __init__(self, threshold=0.7, consec_frames=1, hold_frames=10):
        self.threshold = threshold
        self.consec_frames = consec_frames
        self.hold_frames = hold_frames
        self.streak = 0
        self.hold_counter = 0

    def update_arrays(self, raw_prob, seq=None, mask=None, aspect=1.0) -> bool:
        """Như update() nhưng nhận seq [17,3] / mask [17] của khung hiện tại (None = không thấy người)."""
        if raw_prob >= self.threshold:
            self.streak += 1
        else:
            self.streak = 0

        if self.streak >= self.consec_frames:
            self.hold_counter = self.hold_frames

        if self.hold_counter > 0:
            upright = seq is not None and mask is not None and is_standing_upright(seq, mask, aspect)
            if upright:
                self.hold_counter = max(0, self.hold_counter - 3)
            else:
                self.hold_counter = max(self.hold_counter, 3)
            return True
        return False

    def update(self, raw_prob, current_pose=None, aspect=1.0) -> bool:
        if current_pose is None or not getattr(current_pose, "valid", False):
            return self.update_arrays(raw_prob, None, None, aspect)
        item = from_poses([current_pose])
        return self.update_arrays(raw_prob, item["seq"][0], item["mask"][0], aspect)

    def reset(self):
        self.streak = 0
        self.hold_counter = 0
