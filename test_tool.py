#!/usr/bin/env python3
"""Chạy B0/B1/B2 trên *cùng một* split và lưu dự đoán chi tiết.

Ví dụ:
    python test_tool.py --data data/dataset.npz --split test \
        --b0 models/b0_rule.json --b1 models/best_lstm.pt --b2 models/best_stgcn.pt

Đầu ra gồm ``predictions.csv`` (một dòng / cửa sổ, có nhãn thực tế và
dự đoán của mọi mô hình) và ``summary.json`` (các chỉ số tổng hợp). B1/B2
là tên vai trò trong báo cáo; checkpoint có thể là bất kỳ kiến trúc nào mà
``FallClassifier`` hỗ trợ.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Một số terminal Windows cũ báo cp1252 dù source và thông báo là UTF-8.
# Không để việc in trạng thái làm hỏng một lượt test đã chạy xong.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.b0_rule import B0Rule
from src.train_utils import metrics, metrics_at, roc_auc, video_level, load_splits


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_b0(path: str | Path) -> B0Rule:
    path = _path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model B0: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return B0Rule.from_params(payload.get("params", payload))


def _load_neural(path: str | Path, device: str | None):
    from src.classifier import FallClassifier

    path = _path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    return FallClassifier(path, device=device, verbose=False)


def _latency_ms(predict_one, X: np.ndarray, runs: int = 100) -> float | None:
    """Đo suy luận một cửa sổ sau warm-up; cùng quy ước cho B0/B1/B2."""
    if runs <= 0 or not len(X):
        return None
    one = X[:1]
    for _ in range(min(10, runs)):
        predict_one(one)
    started = time.perf_counter()
    for _ in range(runs):
        predict_one(one)
    return round((time.perf_counter() - started) * 1000.0 / runs, 4)


def _model_result(role: str, model, X: np.ndarray, y: np.ndarray, video: np.ndarray,
                  source: str, threshold: float, pred: np.ndarray, prob: np.ndarray,
                  latency_ms: float | None) -> dict:
    """Tạo một schema báo cáo giống nhau cho rule-based và mạng học sâu."""
    pred = np.asarray(pred, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    return {
        "role": role,
        "source": source,
        "arch": getattr(model, "arch", "rule"),
        "threshold": threshold,
        "latency_ms_per_window": latency_ms,
        "window": metrics(y, pred),
        "score_metrics": metrics_at(y, prob, threshold),
        "auc": roc_auc(y, prob),
        "video": video_level(y, prob, video, threshold),
    }


def run(data_path: str | Path, split: str, b0_path: str | Path | None,
        b1_path: str | Path | None, b2_path: str | Path | None,
        device: str | None = "cpu", latency_runs: int = 100) -> tuple[dict, list[dict]]:
    """Chạy các model đã chọn. Hàm này tách riêng để unit test không cần CLI."""
    dataset_path = _path(data_path)
    data = load_splits(dataset_path)
    X, y, video = data[split]
    if not len(y):
        raise ValueError(f"Split {split!r} rỗng.")

    predictions: dict[str, dict[str, np.ndarray | float]] = {}
    models: list[dict] = []

    if b0_path:
        model = _load_b0(b0_path)
        # B0 là luật nhị phân, vì vậy giữ đúng nhãn do luật quyết định. score()
        # vẫn được lưu để có thể xem độ mạnh/yếu của cảnh báo và tính AUC.
        pred, prob = model.predict(X), model.score(X)
        predictions["b0"] = {"pred": pred, "prob": prob, "threshold": 0.5}
        models.append(_model_result("B0", model, X, y, video, str(b0_path), 0.5, pred, prob,
                                    _latency_ms(model.predict, X, latency_runs)))

    for role, path, key in (("B1", b1_path, "b1"), ("B2", b2_path, "b2")):
        if not path:
            continue
        model = _load_neural(path, device)
        if model.window_size != X.shape[1]:
            raise ValueError(f"{role}: checkpoint cần {model.window_size} khung, "
                             f"dataset có {X.shape[1]} khung.")
        prob = np.asarray(model.predict_batch(X), dtype=np.float64)
        threshold = float(model.fall_threshold)
        pred = (prob >= threshold).astype(np.int64)
        predictions[key] = {"pred": pred, "prob": prob, "threshold": threshold}
        models.append(_model_result(role, model, X, y, video, str(path), threshold, pred, prob,
                                    _latency_ms(model.predict_batch, X, latency_runs)))

    if not predictions:
        raise ValueError("Cần chọn ít nhất một model (--b0, --b1 hoặc --b2).")

    rows = []
    for index, (label, vid) in enumerate(zip(y, video)):
        row = {"split": split, "window_index": index, "video_id": str(vid),
               "actual_label": int(label), "actual_name": "fall" if label else "adl"}
        for key, value in predictions.items():
            pred = int(value["pred"][index])
            row[f"{key}_probability"] = round(float(value["prob"][index]), 6)
            row[f"{key}_prediction"] = pred
            row[f"{key}_prediction_name"] = "fall" if pred else "adl"
        rows.append(row)

    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": {"path": str(data_path), "format": data["format"], "split": split,
                    "windows": int(len(y)), "fall": int(y.sum()), "adl": int((y == 0).sum()),
                    "videos": int(len(set(video.astype(str)))), "window_size": int(X.shape[1]),
                    "fps": data["fps"]},
        "models": models,
    }
    return report, rows


def save(report: dict, rows: list[dict], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = _path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = output_dir / "predictions.csv", output_dir / "summary.json"
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    comparison_path = output_dir / "comparison.csv"
    comparison_rows = []
    for model in report["models"]:
        m = model["window"]
        comparison_rows.append(dict(
            model=model["role"], architecture=model["arch"], threshold=model["threshold"],
            precision=m["prec"], recall=m["rec"], f1=m["f1"], accuracy=m["acc"],
            specificity=m["spec"], auc=model["auc"],
            latency_ms_per_window=model["latency_ms_per_window"],
            tp=m["tp"], fp=m["fp"], fn=m["fn"], tn=m["tn"],
        ))
    with comparison_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return csv_path, json_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Test chung B0/B1/B2 và lưu nhãn + dự đoán.")
    parser.add_argument("--data", default="data/dataset.npz", help="dataset .npz dùng chung")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--b0", default="models/b0_rule.json", help="JSON tham số B0; dùng - để bỏ")
    parser.add_argument("--b1", default="models/best_lstm.pt", help="checkpoint B1; dùng - để bỏ")
    parser.add_argument("--b2", default="models/best_stgcn.pt", help="checkpoint B2; dùng - để bỏ")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0")
    parser.add_argument("--latency-runs", type=int, default=100,
                        help="số lượt đo thời gian một cửa sổ (0 = bỏ qua)")
    parser.add_argument("--out", default=None, help="thư mục kết quả (mặc định output/test_runs/<thời gian>)")
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or f"output/test_runs/{stamp}"
    try:
        selected = lambda value: None if value in ("", "-") else value
        report, rows = run(args.data, args.split, selected(args.b0), selected(args.b1),
                           selected(args.b2), args.device, args.latency_runs)
        csv_path, json_path = save(report, rows, out)
    except Exception as exc:
        parser.error(str(exc))
    print(f"Đã kiểm thử {len(report['models'])} model trên {report['dataset']['windows']} cửa sổ của split {args.split}.")
    for model in report["models"]:
        m = model["window"]
        print(f"  {model['role']}: F1={m['f1']:.3f}, P={m['prec']:.3f}, R={m['rec']:.3f}, "
              f"{model['latency_ms_per_window']} ms/cửa sổ")
    print(f"Dự đoán: {csv_path}")
    print(f"Tổng hợp: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
