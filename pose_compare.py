#!/usr/bin/env python3
"""So sánh hai YOLO-Pose mà không thay đổi model mặc định của web.

Mỗi model được trích skeleton vào thư mục riêng, sau đó cùng B0/B1/B2 chạy
trên các cửa sổ đúng cùng video/thời điểm. Nếu cung cấp manifest CSV có các
cột ``video_id,path,fall_start_s,fall_end_s`` thì file tổng hợp cũng có nhãn
thực tế; nếu không, công cụ vẫn báo chất lượng pose và dự đoán để rà soát.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.b0_rule import B0Rule
from src.features import normalize_batch
from src.skeleton_io import make_windows, seconds_to_frames
from web.config import load_config
from web.engine import build_estimator, extract_to_csv, load_npz

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}


def _videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def _manifest(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not {"video_id", "path"} <= set(rows[0] if rows else []):
        raise ValueError("manifest cần cột video_id,path (fall_start_s/fall_end_s là tùy chọn).")
    return {str(r["video_id"]): r for r in rows}


def _load_classifiers(b0: str | None, b1: str | None, b2: str | None):
    out = {}
    if b0:
        payload = json.loads(Path(b0).read_text(encoding="utf-8"))
        out["b0"] = B0Rule.from_params(payload.get("params", payload))
    if b1 or b2:
        from src.classifier import FallClassifier
        for key, path in (("b1", b1), ("b2", b2)):
            if path:
                out[key] = FallClassifier(path, device="cpu", verbose=False)
    return out


def _predictions(classifiers: dict, X: np.ndarray) -> dict[str, np.ndarray]:
    result = {}
    for key, model in classifiers.items():
        if key == "b0":
            result[key] = model.predict(X)
        else:
            if model.window_size != X.shape[1]:
                raise ValueError(f"{key.upper()} cần {model.window_size} frame, nhận {X.shape[1]}.")
            result[key] = (model.predict_batch(X) >= model.fall_threshold).astype(int)
    return result


def run(args) -> tuple[dict, list[dict]]:
    root = Path(args.videos).resolve()
    manifest = _manifest(args.manifest)
    videos = _videos(root) if root.is_dir() else [root]
    if not videos:
        raise ValueError("Không tìm thấy video đầu vào.")
    classifiers = _load_classifiers(args.b0, args.b1, args.b2)
    cfg = load_config(args.config, [])
    runs, rows = {}, []
    for name, checkpoint in (("baseline", args.baseline), ("candidate", args.candidate)):
        local = copy.deepcopy(cfg)
        local["pose"]["model_path"] = str(Path(checkpoint).resolve())
        local["pose"]["device"] = args.device
        estimator = build_estimator(local)
        summary, windows = [], {}
        try:
            for video in videos:
                vid = video.stem
                out = Path(args.out) / name
                stat = extract_to_csv(str(video), vid, out / "csv" / f"{vid}.csv", local,
                                      estimator=estimator, out_npz=out / "npz" / f"{vid}.npz")
                summary.append({"video_id": vid, **stat})
                item = load_npz(out / "npz" / f"{vid}.npz")
                item["video_id"] = vid
                label = manifest.get(vid, {})
                intervals = []
                try:
                    a, b = float(label["fall_start_s"]), float(label["fall_end_s"])
                    if b > a:
                        intervals = [(seconds_to_frames(a, item["fps"]), seconds_to_frames(b, item["fps"]))]
                except (KeyError, TypeError, ValueError):
                    pass
                X, M, y, _ = make_windows(item, win_sec=args.window_sec, stride_sec=args.stride_sec,
                                           fall_intervals=intervals, min_overlap=args.min_overlap)
                windows[vid] = (normalize_batch(X, M, item["aspect"]), y)
        finally:
            estimator.close()
        runs[name] = {"summary": summary, "windows": windows}

    # Cùng vị trí window: hai lượt trích dùng cùng FPS/cửa sổ. Không có nhãn thì actual_label để trống.
    for vid in sorted(set(runs["baseline"]["windows"]) & set(runs["candidate"]["windows"])):
        bx, by = runs["baseline"]["windows"][vid]
        cx, cy = runs["candidate"]["windows"][vid]
        n = min(len(bx), len(cx))
        bp, cp = _predictions(classifiers, bx[:n]), _predictions(classifiers, cx[:n])
        labelled = vid in manifest and (manifest[vid].get("fall_start_s") not in (None, ""))
        for i in range(n):
            row = {"video_id": vid, "window_index": i,
                   "actual_label": int(by[i]) if labelled else ""}
            for key in classifiers:
                row[f"{key}_baseline"] = int(bp[key][i])
                row[f"{key}_candidate"] = int(cp[key][i])
                row[f"{key}_changed"] = int(bp[key][i] != cp[key][i])
            rows.append(row)
    report = {"videos": len(videos), "windows_compared": len(rows), "labelled": bool(manifest),
              "pose": {k: v["summary"] for k, v in runs.items()},
              "classifiers": list(classifiers)}
    return report, rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="So sánh pose baseline/candidate qua B0/B1/B2.")
    p.add_argument("--videos", default="data/raw")
    p.add_argument("--baseline", default="models/yolo11n-pose.pt")
    p.add_argument("--candidate", required=True)
    p.add_argument("--manifest", help="CSV: video_id,path,fall_start_s,fall_end_s")
    p.add_argument("--config", default="configs/web.yaml")
    p.add_argument("--out", default="output/pose_compare")
    p.add_argument("--device", default="cpu")
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument("--stride-sec", type=float, default=0.5)
    p.add_argument("--min-overlap", type=float, default=0.3)
    p.add_argument("--b0", default="models/b0_rule.json")
    p.add_argument("--b1", default="models/best_lstm.pt")
    p.add_argument("--b2", default="models/best_stgcn.pt")
    args = p.parse_args(argv)
    try:
        report, rows = run(args)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with (out / "classifier_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["video_id"])
            w.writeheader(); w.writerows(rows)
    except Exception as exc:
        p.error(str(exc))
    print(f"Đã so sánh {report['videos']} video, {report['windows_compared']} cửa sổ. Kết quả: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
