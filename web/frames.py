"""Xuất toàn bộ khung của một video thành một file ZIP, mỗi khung đã vẽ skeleton.

Skeleton lấy từ CSV đã trích (đúng dữ liệu đang hiển thị trên trình phát), nên không chạy lại model.
Khung nào không có dòng tương ứng trong CSV vẫn được ghi, nhưng không có skeleton.
Khung được đọc bằng cùng cách lấy mẫu như lúc trích (target_fps), nên chỉ số khớp với frame_idx trong CSV.
"""
from __future__ import annotations

import zipfile

import cv2
import numpy as np

from src.pose_estimate import PoseResult

from .csv_tools import read_csv, to_arrays
from .engine import OpenedSource, build_drawer, draw_hud


def build_frames_zip(spec: str, video_id: str, csv_path, cfg: dict, dest) -> int:
    """Ghi ZIP vào dest (đường dẫn hoặc file-like như BytesIO). Trả về số khung đã ghi."""
    df = read_csv(csv_path)
    t_csv, valid, joints = to_arrays(df)
    row_of = {int(f): k for k, f in enumerate(df["frame_idx"].to_numpy())}
    drawer = build_drawer(cfg)
    n = 0
    with OpenedSource(spec, cfg, video_id=video_id) as src, \
            zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as zf:
        for pkt in src.frames():
            h, w = pkt.frame.shape[:2]
            k = row_of.get(pkt.idx)
            pose = None if k is None else PoseResult(
                pkt.idx, float(t_csv[k]), bool(valid[k]), joints[k].astype(np.float32), w, h)
            img = draw_hud(drawer.draw(pkt.frame, pose), [f"khung {pkt.idx}  t={pkt.t:.2f}s"])
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise IOError(f"không mã hóa được khung {pkt.idx}")
            zf.writestr(f"frame_{pkt.idx:06d}_t{pkt.t:.2f}.jpg", buf.tobytes())
            n += 1
    if n == 0:
        raise ValueError("không đọc được khung nào từ video")
    return n
