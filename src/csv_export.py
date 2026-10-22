"""
csv_export.py
-------------
Xuất PoseResult ra CSV, MỖI KHUNG MỘT DÒNG (kể cả khung không có người).

Cột: video_id, frame_idx, t, valid, rồi <joint>_x, <joint>_y, <joint>_conf
cho 17 khớp. x, y chuẩn hóa 0-1; khớp thiếu / khung valid=0 để trống.

Giả định: schema này khớp skeleton_io.py của FallApp. Nếu FallApp dùng
tên/thứ tự cột khác, chỉ cần sửa META_COLUMNS và _joint_columns().
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from .constants import KEYPOINT_NAMES
from .pose_estimate import PoseResult

META_COLUMNS = ["video_id", "frame_idx", "t", "valid"]


def _joint_columns():
    cols = []
    for name in KEYPOINT_NAMES:
        cols += [f"{name}_x", f"{name}_y", f"{name}_conf"]
    return cols


def _fmt(v, nd):
    return "" if v is None or not math.isfinite(v) else round(float(v), nd)


class CSVExporter:
    def __init__(self, video_id, output_dir="output"):
        self.video_id = Path(str(video_id)).stem
        self.output_path = Path(output_dir) / "csv" / f"{self.video_id}.csv"
        self.file = None
        self.writer = None
        self.total = 0        # số dòng đã ghi
        self.invalid = 0      # số dòng valid=0
        self.gaps = 0         # số frame_idx bị nhảy cóc (không có dòng)
        self._last_idx = -1

    def open(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output_path.open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(
            self.file, fieldnames=META_COLUMNS + _joint_columns()
        )
        self.writer.writeheader()
        self.total = self.invalid = self.gaps = 0
        self._last_idx = -1
        return self

    def write(self, pose: PoseResult):
        if self.writer is None:
            raise RuntimeError("Hãy gọi open() trước.")

        if pose.frame_idx > self._last_idx + 1:
            self.gaps += pose.frame_idx - self._last_idx - 1
        self._last_idx = pose.frame_idx

        row = {
            "video_id": self.video_id,
            "frame_idx": pose.frame_idx,
            "t": _fmt(pose.t, 3),
            "valid": int(bool(pose.valid)),
        }
        for i, name in enumerate(KEYPOINT_NAMES):
            if pose.valid:
                x, y, c = pose.joints[i]
            else:
                x = y = c = None
            missing = x is None or not math.isfinite(x) or not math.isfinite(y)
            row[f"{name}_x"] = "" if missing else _fmt(x, 5)
            row[f"{name}_y"] = "" if missing else _fmt(y, 5)
            row[f"{name}_conf"] = "" if missing else _fmt(c, 4)

        self.writer.writerow(row)
        self.total += 1
        if not pose.valid:
            self.invalid += 1

    write_frame = write  # tên cũ

    @property
    def missing_ratio(self) -> float:
        """Tỉ lệ khung mất = (khung valid=0 + khung bị nhảy cóc) / tổng khung."""
        n = self.total + self.gaps
        return (self.invalid + self.gaps) / n if n else 0.0

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()