"""
postprocess.py
--------------
Xử lý offline trên CSV đã xuất: nội suy tuyến tính các khoảng trống NGẮN của
từng khớp (có quan sát thật ở cả hai đầu). Khoảng trống dài hơn max_gap giữ
nguyên (để trống) vì nội suy lâu là bịa dữ liệu.

- Khớp "giữ tạm" của SkeletonFilter (conf = 0 nhưng x, y có số) được coi là
  thiếu, vì đó là vị trí đóng băng, không phải quan sát.
- Ô được nội suy có conf = 0 để phía sau phân biệt được với quan sát thật.
- Cột valid giữ nguyên; thêm cột interp_count = số khớp được nội suy ở khung đó.
- Ghi ra tệp mới <video_id>_interp.csv, không đụng tệp gốc.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .constants import KEYPOINT_NAMES


def _to_float(v):
    try:
        return float(v) if v not in ("", None) else np.nan
    except ValueError:
        return np.nan


def _fill_short_gaps(a: np.ndarray, max_gap: int) -> np.ndarray:
    """Trả mask các vị trí được nội suy (a sửa tại chỗ)."""
    filled = np.zeros(len(a), dtype=bool)
    nan = np.isnan(a)
    i, n = 0, len(a)
    while i < n:
        if not nan[i]:
            i += 1
            continue
        j = i
        while j < n and nan[j]:
            j += 1
        # khoảng trống [i, j): cần quan sát thật ở i-1 và j
        if i > 0 and j < n and (j - i) <= max_gap:
            a[i:j] = np.interp(np.arange(i, j), [i - 1, j], [a[i - 1], a[j]])
            filled[i:j] = True
        i = j
    return filled


def interpolate_csv(csv_path, max_gap: int = 6, out_path=None):
    csv_path = Path(csv_path)
    out_path = Path(out_path) if out_path else csv_path.with_name(csv_path.stem + "_interp.csv")

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    if not rows:
        return out_path, 0

    n = len(rows)
    interp_count = np.zeros(n, dtype=int)
    total_filled = 0

    for name in KEYPOINT_NAMES:
        x = np.array([_to_float(r[f"{name}_x"]) for r in rows])
        y = np.array([_to_float(r[f"{name}_y"]) for r in rows])
        c = np.array([_to_float(r[f"{name}_conf"]) for r in rows])

        # vị trí đóng băng (conf = 0) không phải quan sát thật
        frozen = np.nan_to_num(c, nan=0.0) <= 0
        x[frozen] = np.nan
        y[frozen] = np.nan

        had_value = ~np.isnan(x) & ~np.isnan(y)
        filled = _fill_short_gaps(x, max_gap) & _fill_short_gaps(y, max_gap)
        # chỉ nhận ô nội suy khi cả x lẫn y cùng được lấp
        filled = filled & ~np.isnan(x) & ~np.isnan(y)
        # khung bị giữ tạm mà không được nội suy: để trống cho nhất quán
        for k, r in enumerate(rows):
            if filled[k]:
                r[f"{name}_x"] = round(float(x[k]), 5)
                r[f"{name}_y"] = round(float(y[k]), 5)
                r[f"{name}_conf"] = 0
            elif frozen[k] and not had_value[k]:
                r[f"{name}_x"] = r[f"{name}_y"] = r[f"{name}_conf"] = ""
        interp_count += filled.astype(int)
        total_filled += int(filled.sum())

    for k, r in enumerate(rows):
        r["interp_count"] = int(interp_count[k])
    if "interp_count" not in fields:
        fields.append("interp_count")

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out_path, total_filled