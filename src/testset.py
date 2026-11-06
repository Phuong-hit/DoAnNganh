"""
testset.py
----------
BỘ TEST CHUNG của nhóm: một thư mục cố định, mọi mô hình (B0 luật, B1 LSTM, B2 Shift-GCN/ST-GCN, ...)
đều đánh giá trên ĐÚNG dữ liệu này, cùng nhãn, cùng định dạng.

    data/testsets/<tên>/
        testset.npz     X [N,T,17,3] float32, y [N] int64, video [N] str, subject [N] str
        manifest.json   nguồn, quy ước nhãn/định dạng, số lượng, danh sách subject, mã băm nội dung
        videos.csv      mỗi video một dòng: subject, loại, địa điểm, hành động, hướng, nhãn, số cửa sổ

Quy ước (giống dataset huấn luyện, xem build_dataset_v1.py và src/features.py):
    X[..., 0:2]  tọa độ x, y đã chuẩn hóa bằng features.normalize_clip (tâm hông, chia độ dài xương trung vị)
    X[..., 2]    mask: 1 = khớp quan sát thật, 0 = thiếu (x = y = 0)
    y            0 = KHÔNG NGÃ (ADL), 1 = NGÃ (Fall)
    video        "SBJ13_Fall_LOC2-ACT10-F-1" = <subject>_<loại>_<địa điểm>-<hành động>-<hướng>-<lần>

load_testset() kiểm tra lại mã băm: ai sửa/ghi đè dữ liệu test sẽ bị báo ngay, để kết quả các thành viên
luôn so sánh được với nhau.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
LABELS = {0: "ADL (không ngã)", 1: "Fall (ngã)"}
VIDEO_RE = re.compile(r"^(?P<subject>[^_]+)_(?P<kind>[^_]+)_(?:(?P<loc>LOC\d+)-)?(?P<act>ACT\d+)?-?"
                      r"(?P<view>[A-Z])?-?(?P<trial>\d+)?", re.IGNORECASE)


def parse_video_id(video_id: str) -> dict:
    """'SBJ13_Fall_LOC2-ACT10-F-1' -> {subject, kind, loc, act, view, trial}. Phần không nhận ra = ''."""
    m = VIDEO_RE.match(str(video_id))
    out = {k: "" for k in ("subject", "kind", "loc", "act", "view", "trial")}
    if m:
        out.update({k: (v or "") for k, v in m.groupdict().items()})
    else:
        out["subject"] = str(video_id).split("_")[0]
    return out


def content_hash(X, y, video) -> str:
    """SHA-256 của nội dung (X, y, video) — không phụ thuộc cách nén của file npz."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    h.update("\n".join(map(str, video)).encode("utf-8"))
    return h.hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_testset(data_path, out_dir, split: str = "test", name: str | None = None,
                   note: str = "", overwrite: bool = False) -> Path:
    """Tách một split của dataset huấn luyện thành bộ test chung (testset.npz + manifest.json + videos.csv)."""
    from .train_utils import load_splits, subject_of

    data_path, out_dir = Path(data_path), Path(out_dir)
    if (out_dir / "testset.npz").exists() and not overwrite:
        raise FileExistsError(f"{out_dir} đã có bộ test. Bộ test chung không nên ghi đè; "
                              f"đặt tên mới (vd v2_test) hoặc dùng overwrite=True nếu chắc chắn.")
    d = load_splits(data_path)
    X, y, video = d[split]
    if not len(y):
        raise ValueError(f"Split {split!r} rỗng.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Split {split!r} phải có cả hai lớp, có {sorted(set(np.unique(y)))}.")
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    video = np.asarray(video).astype(str)
    subject = np.array([subject_of(v) for v in video])

    # Mỗi video chỉ được có MỘT nhãn video (có ít nhất một cửa sổ ngã = video ngã) và tên loại phải khớp nhãn
    videos = []
    for v in sorted(set(video)):
        m = video == v
        info = parse_video_id(v)
        lab = int(y[m].max())
        videos.append(dict(video_id=v, **info, label=lab, n_windows=int(m.sum()), n_fall_windows=int(y[m].sum())))
    mismatch = [r["video_id"] for r in videos
                if r["kind"].lower() in ("fall", "adl") and (r["kind"].lower() == "fall") != bool(r["label"])]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "testset.npz", X=X, y=y, video=video, subject=subject)
    with open(out_dir / "videos.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(videos[0]))
        w.writeheader()
        w.writerows(videos)

    manifest = dict(
        name=name or out_dir.name, format_version=FORMAT_VERSION,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"), note=note,
        source=dict(path=str(data_path).replace("\\", "/"), split=split, sha256=file_hash(data_path),
                    format=d["format"]),
        window_size=int(X.shape[1]), fps=float(d["fps"]), num_keypoints=int(X.shape[2]),
        channels=["x (đã chuẩn hóa)", "y (đã chuẩn hóa)", "mask (1 = khớp quan sát)"],
        normalization="src/features.py normalize_clip: một tâm (trung bình hai hông) và một tỉ lệ "
                      "(độ dài xương trung vị) cho cả cửa sổ; khớp thiếu = 0",
        labels={str(k): v for k, v in LABELS.items()},
        video_label_rule="video là NGÃ nếu có ít nhất một cửa sổ nhãn 1; dự đoán video là NGÃ nếu có ít nhất "
                         "một cửa sổ được dự đoán NGÃ (giống cảnh báo thật)",
        counts=dict(windows=int(len(y)), fall_windows=int(y.sum()), adl_windows=int((y == 0).sum()),
                    videos=len(videos), fall_videos=sum(r["label"] for r in videos),
                    adl_videos=sum(1 - r["label"] for r in videos)),
        subjects=sorted(set(subject.tolist())),
        label_kind_mismatch=mismatch,
        content_sha256=content_hash(X, y, video),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def is_testset(path) -> bool:
    p = Path(path)
    return (p.is_dir() and (p / "testset.npz").is_file()) or (p.name == "testset.npz" and p.is_file())


def load_testset(path, verify: bool = True) -> dict:
    """Thư mục bộ test (hoặc đường dẫn testset.npz) -> {"X", "y", "video", "subject", "manifest"}."""
    p = Path(path)
    d = p if p.is_dir() else p.parent
    with np.load(d / "testset.npz", allow_pickle=False) as z:
        X, y = z["X"].astype(np.float32), z["y"].astype(np.int64)
        video = z["video"].astype(str)
        subject = z["subject"].astype(str) if "subject" in z.files else np.array([v.split("_")[0] for v in video])
    mpath = d / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {}
    if verify and manifest.get("content_sha256"):
        got = content_hash(X, y, video)
        if got != manifest["content_sha256"]:
            raise ValueError(f"Bộ test {d} đã bị thay đổi (mã băm {got[:12]}… khác manifest "
                             f"{manifest['content_sha256'][:12]}…). Lấy lại bản gốc hoặc tạo bộ test mới.")
    if X.ndim != 4 or X.shape[2:] != (17, 3) or len(X) != len(y) or len(y) != len(video):
        raise ValueError(f"Bộ test {d} sai định dạng: X {X.shape}, y {y.shape}, video {video.shape}.")
    return dict(X=X, y=y, video=video, subject=subject, manifest=manifest, path=str(d))
