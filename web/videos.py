"""Danh sách video: quét data/videos/{fall,adl}/ -> data/metadata.csv (giống fall_app).

Các cột nhập tay (subject_id, scene, fall_start_s, fall_end_s, split, note...) được GIỮ NGUYÊN
khi quét lại. Chỉ hỗ trợ FILE video, vì FileSource của src/ chỉ mở được file video
(không đọc thư mục ảnh như fall_app).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from .config import ROOT
from .engine import VIDEO_EXTS, make_video_id

COLUMNS = [
    "video_id", "path", "dataset", "subject_id", "scene", "camera", "label", "activity",
    "fall_start_s", "fall_end_s", "fps", "width", "height", "frame_count", "duration_s",
    "split", "note",
]
VALID_LABELS = {"fall", "adl"}
VALID_SPLITS = {"train", "val", "test", ""}
_STR_COLS = {c: str for c in ("video_id", "subject_id", "scene", "camera", "split", "note", "activity", "dataset")}


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return dict(fps=0, width=0, height=0, frame_count=0, duration_s=0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    d = dict(fps=round(fps, 3), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), frame_count=n,
             duration_s=round(n / fps, 3) if fps else 0)
    cap.release()
    return d


def load_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=_STR_COLS)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df


def scan_videos(videos_dir: str | Path, metadata_path: str | Path, dataset: str = "self") -> tuple[pd.DataFrame, list[str]]:
    """Quét videos_dir/<label>/ (label = fall | adl). Trả (bảng metadata, danh sách cảnh báo)."""
    videos_dir, metadata_path = Path(videos_dir), Path(metadata_path)
    notes: list[str] = []
    old = load_metadata(metadata_path).set_index("video_id", drop=False)
    dup = old.index.duplicated(keep="first")
    if dup.any():
        notes.append(f"metadata.csv có video_id bị trùng: {list(old.index[dup])} — giữ dòng đầu tiên")
        old = old[~dup]
    rows, seen = [], set()
    videos_dir.mkdir(parents=True, exist_ok=True)
    for label_dir in sorted(p for p in videos_dir.iterdir() if p.is_dir()):
        label = label_dir.name.lower()
        if label not in VALID_LABELS:
            notes.append(f"Bỏ qua thư mục {label_dir.name}: tên phải là fall hoặc adl")
            continue
        for item in sorted(label_dir.iterdir()):
            if not (item.is_file() and item.suffix.lower() in VIDEO_EXTS):
                if item.is_dir():
                    notes.append(f"Bỏ qua thư mục {item.name}: chỉ hỗ trợ file video")
                continue
            vid = make_video_id(item.stem)
            if vid in seen:
                raise ValueError(f"Trùng video_id '{vid}' ({item}). Đổi tên file cho khác nhau "
                                 "(CSV đặt tên theo video_id nên trùng sẽ ghi đè nhau).")
            seen.add(vid)
            res = item.resolve()
            rel = res.relative_to(ROOT) if res.is_relative_to(ROOT) else res
            auto = probe_video(item)
            if vid in old.index:
                row = old.loc[vid].to_dict()
                row.update({"path": str(rel), **auto})
                if not row.get("label") or pd.isna(row.get("label")):
                    row["label"] = label
            else:
                row = {c: None for c in COLUMNS}
                row.update(video_id=vid, path=str(rel), dataset=dataset, label=label,
                           fall_start_s=-1.0, fall_end_s=-1.0, split="", note="", **auto)
            rows.append(row)
    gone = sorted(set(old.index) - seen)
    for vid in gone:
        r = old.loc[vid].to_dict()
        note = r.get("note")
        note = "" if note is None or pd.isna(note) else str(note)
        if "[MISSING FILE]" not in note:
            r["note"] = (note + " [MISSING FILE]").strip()
        rows.append(r)
    df = pd.DataFrame(rows, columns=COLUMNS)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False)
    if gone:
        notes.append(f"{len(gone)} video có trong metadata nhưng không còn file: {gone[:5]}")
    return df, notes + validate(df)


def validate(df: pd.DataFrame) -> list[str]:
    """Lỗi/cảnh báo cần sửa tay trong metadata.csv (nhãn, thời điểm ngã, rò rỉ theo người)."""
    problems = []
    if df.empty:
        return problems
    for r in df.itertuples():
        fs = float(r.fall_start_s) if pd.notna(r.fall_start_s) else -1
        fe = float(r.fall_end_s) if pd.notna(r.fall_end_s) else -1
        if r.label not in VALID_LABELS:
            problems.append(f"{r.video_id}: label '{r.label}' không hợp lệ")
        if r.label == "fall" and fs < 0:
            problems.append(f"{r.video_id}: video ngã chưa có fall_start_s/fall_end_s")
        if r.label == "fall" and fs >= 0 and fe <= fs:
            problems.append(f"{r.video_id}: fall_end_s phải lớn hơn fall_start_s")
        if r.label == "adl" and fs >= 0:
            problems.append(f"{r.video_id}: video ADL nhưng có thời điểm ngã")
        dur = float(r.duration_s) if pd.notna(r.duration_s) and r.duration_s else None
        if dur and fe > dur + 0.5:
            problems.append(f"{r.video_id}: fall_end_s={fe} vượt thời lượng {dur}s")
        split = "" if pd.isna(r.split) else str(r.split)
        if split not in VALID_SPLITS:
            problems.append(f"{r.video_id}: split '{split}' phải là train/val/test")
        if "[MISSING FILE]" in str(r.note):
            problems.append(f"{r.video_id}: không tìm thấy file video")
    has = df[df.subject_id.notna() & (df.split.fillna("") != "")]
    if len(has):
        leak = has.groupby("subject_id").split.nunique()
        for sid in leak[leak > 1].index:
            problems.append(f"Rò rỉ dữ liệu: subject {sid} nằm ở nhiều tập "
                            f"{sorted(has[has.subject_id == sid].split.unique())}")
    return problems
