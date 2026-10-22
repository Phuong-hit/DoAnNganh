"""Đọc / kiểm tra / chuyển dạng file CSV do src/csv_export.py (CSVExporter) sinh ra.

Quy ước của CSVExporter (khác fall_app!):
  - cột: video_id, frame_idx, t, valid, <khớp>_x, <khớp>_y, <khớp>_conf (17 khớp, thứ tự COCO)
  - khung valid=0 và khớp không quan sát được: ô ĐỂ TRỐNG (đọc vào thành NaN), không phải 0
  - khớp do SkeletonFilter "giữ tạm": có x, y nhưng conf = 0

Vì vậy không dùng lại được sequence_check.py của fall_app (nó coi NaN là lỗi và đòi valid=0 phải
toàn 0). Bản kiểm tra dưới đây giữ nguyên các mục của fall_app nhưng theo quy ước trên.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import KEYPOINT_NAMES, NUM_KEYPOINTS
from src.csv_export import META_COLUMNS, _joint_columns

JOINT_COLUMNS = _joint_columns()          # lấy thẳng từ src/ để không bao giờ lệch tên cột


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def to_arrays(df: pd.DataFrame):
    """-> t [T], valid [T] bool, joints [T,17,3] float (NaN = trống)."""
    T = len(df)
    joints = df[JOINT_COLUMNS].to_numpy(dtype=np.float64).reshape(T, NUM_KEYPOINTS, 3)
    return df["t"].to_numpy(dtype=np.float64), df["valid"].to_numpy() != 0, joints


def to_json_payload(path: str | Path) -> dict:
    """Dữ liệu cho lớp vẽ trên trình phát: NaN -> null."""
    t, valid, joints = to_arrays(read_csv(path))
    j = np.round(joints, 4).astype(object)
    j[~np.isfinite(joints)] = None
    return dict(t=np.round(t, 3).tolist(), valid=valid.astype(int).tolist(), joints=j.tolist())


def to_long(path: str | Path, out_path: str | Path) -> Path:
    """CSV dạng rộng (1 dòng/khung) -> dạng dài (1 dòng/khớp): video_id, frame_idx, t, valid, kpt_id, kpt_name, x, y, conf."""
    df = read_csv(path)
    T = len(df)
    base = df[META_COLUMNS]
    rep = base.loc[base.index.repeat(NUM_KEYPOINTS)].reset_index(drop=True)
    rep["kpt_id"] = np.tile(np.arange(NUM_KEYPOINTS), T)
    rep["kpt_name"] = np.tile(KEYPOINT_NAMES, T)
    flat = df[JOINT_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 3)
    rep["x"], rep["y"], rep["conf"] = flat[:, 0], flat[:, 1], flat[:, 2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# ---------------------------------------------------------------------------- kiểm tra cho xử lý chuỗi
class Report:
    def __init__(self, name: str):
        self.name = name
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: dict = {}

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return dict(ok=self.ok, errors=self.errors, warnings=self.warnings, info=self.info)

    def to_text(self) -> str:
        lines = [f"{'ĐẠT' if self.ok else 'LỖI'} {self.name}"]
        if self.info:
            lines.append("  " + " | ".join(f"{k}={v}" for k, v in self.info.items()))
        lines += [f"  [LỖI] {m}" for m in self.errors] + [f"  [CẢNH BÁO] {m}" for m in self.warnings]
        return "\n".join(lines)


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def check_csv(path: str | Path, ccfg: dict | None = None) -> Report:
    path = Path(path)
    ccfg = ccfg or {}
    window_s = float(ccfg.get("window_s", 2.0))
    stride_s = float(ccfg.get("stride_s", 0.25))
    conf_thres = float(ccfg.get("kpt_conf_thres", 0.2))
    max_missing = float(ccfg.get("max_missing", 0.4))
    rep = Report(path.name)

    try:
        df = read_csv(path)
    except Exception as e:
        rep.errors.append(f"không đọc được file: {e}")
        return rep

    missing_cols = [c for c in META_COLUMNS + JOINT_COLUMNS if c not in df.columns]
    if missing_cols:
        rep.errors.append(f"thiếu {len(missing_cols)} cột, ví dụ {missing_cols[:5]} — "
                          "file có phải do CSVExporter (src/csv_export.py) sinh ra không?")
        return rep
    T = len(df)
    rep.info["frames"] = T
    if T < 2:
        rep.errors.append("file có ít hơn 2 dòng, không tạo được chuỗi")
        return rep
    try:
        t, valid, joints = to_arrays(df)
    except (ValueError, TypeError) as e:
        rep.errors.append(f"không dựng được mảng [T,17,3]: {e}")
        return rep

    # --- thời gian ---
    if not np.isfinite(t).all():
        rep.errors.append("cột t có ô trống/NaN")
        return rep
    dt = np.diff(t)
    if (dt <= 0).any():
        rep.errors.append(f"{int((dt <= 0).sum())} chỗ t không tăng (trùng hoặc lùi) — không cắt cửa sổ theo thời gian được")
        return rep
    med = float(np.median(dt))
    rep.info["fps_suy_ra"] = round(1 / med, 2)
    rep.info["thoi_luong_s"] = round(float(t[-1] - t[0]), 2)
    jitter = float(np.max(np.abs(dt - med)) / med)
    rep.info["lech_nhip_max"] = f"{jitter * 100:.0f}%"
    if jitter > 0.5:
        rep.warnings.append(f"khoảng cách giữa các mẫu lệch tới {jitter * 100:.0f}% so với trung vị "
                            f"({med * 1000:.0f} ms) — đặc trưng vận tốc có thể sai (thường gặp với webcam xử lý chậm)")

    # --- frame_idx liên tục (CSVExporter đếm khung bị nhảy cóc vào missing_ratio) ---
    fi = df["frame_idx"].to_numpy()
    if not np.array_equal(fi, np.arange(fi[0], fi[0] + T)):
        rep.warnings.append("frame_idx không liên tục — có khung bị bỏ hoặc file bị ghép/lọc")

    # --- quy ước ô trống ---
    if (~valid).any() and np.isfinite(joints[~valid]).any():
        rep.errors.append("có khung valid=0 nhưng ô keypoint không trống — sai quy ước của CSVExporter")
    xy_fin = np.isfinite(joints[:, :, :2])
    if (xy_fin[:, :, 0] != xy_fin[:, :, 1]).any():
        rep.errors.append("có khớp chỉ có x hoặc chỉ có y — file bị hỏng")

    if not valid.any():
        rep.errors.append("không có khung nào phát hiện được người")
        return rep
    jv = joints[valid]
    obs = np.isfinite(jv[:, :, 0])
    xy = jv[:, :, :2][obs]
    if xy.size:
        lo, hi = float(xy.min()), float(xy.max())
        if lo < -0.001 or hi > 1.001:
            rep.errors.append(f"tọa độ ngoài 0..1 (min={lo:.3f}, max={hi:.3f}) — có thể còn là pixel")
    conf = jv[:, :, 2]
    held = obs & (conf == 0)
    rep.info["khop_thieu"] = f"{(1 - obs.mean()) * 100:.1f}%"
    if held.any():
        rep.info["khop_giu_tam"] = int(held.sum())
    if 1 - obs.mean() > 0.3:
        rep.warnings.append(f"{(1 - obs.mean()) * 100:.0f}% khớp trống ở các khung có người — "
                            "cần nội suy/điền trước khi đưa vào mô hình chuỗi")

    # --- mất người ---
    miss = float(1 - valid.mean())
    rep.info["ti_le_mat_nguoi"] = f"{miss * 100:.1f}%"
    gap_s = _longest_run(~valid) * med
    rep.info["khoang_trong_dai_nhat_s"] = round(gap_s, 2)
    if miss > max_missing:
        rep.warnings.append(f"mất người {miss * 100:.0f}% số khung, vượt max_missing={max_missing * 100:.0f}%")
    if gap_s > window_s:
        rep.warnings.append(f"có đoạn mất người liên tiếp {gap_s:.1f}s, dài hơn cửa sổ {window_s}s — nội suy không đáng tin")

    seen_conf = conf[obs & (conf > 0)]
    mean_conf = float(seen_conf.mean()) if seen_conf.size else 0.0
    rep.info["conf_tb"] = round(mean_conf, 3)
    good = (np.nan_to_num(joints[:, :, 2]) >= conf_thres).sum(1)
    rep.info["kpt_tot_tb"] = round(float(good[valid].mean()), 1)
    if mean_conf < 0.3:
        rep.warnings.append(f"confidence trung bình thấp ({mean_conf:.2f}) — kiểm tra lại model hoặc video")

    # --- cửa sổ ---
    duration = float(t[-1] - t[0])
    if duration < window_s:
        rep.errors.append(f"video chỉ dài {duration:.1f}s, ngắn hơn cửa sổ {window_s}s")
    else:
        n_win = usable = 0
        start = t[0]
        while start + window_s <= t[-1] + 1e-6:
            m = (t >= start) & (t < start + window_s)
            if m.sum() >= 3:
                n_win += 1
                if 1 - valid[m].mean() <= max_missing:
                    usable += 1
            start += stride_s
        rep.info["cua_so"] = f"{usable}/{n_win} dùng được"
        if usable == 0:
            rep.errors.append(f"không cửa sổ nào đạt (mọi cửa sổ đều mất người quá {max_missing * 100:.0f}%)")
        elif usable < n_win * 0.5:
            rep.warnings.append(f"chỉ {usable}/{n_win} cửa sổ dùng được")

    if df["video_id"].nunique() > 1:
        rep.errors.append(f"file chứa {df['video_id'].nunique()} video_id — mỗi file phải là một video")
    return rep
