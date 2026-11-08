#!/usr/bin/env python3
"""
test_tool.py
------------
Test Tool chung: chạy B0 (luật) / B1 (LSTM) / B2 (Shift-GCN, ST-GCN, ...) trên CÙNG một bộ test, lưu
nhãn thực tế + dự đoán của từng model, tổng hợp Precision / Recall / F1 / thời gian xử lý, và liệt kê các
trường hợp BÁO NHẦM (FP) và BỎ SÓT (FN) để phân tích.

1) Tạo bộ test chung (một lần, cả nhóm dùng chung; không ghi đè):
    python test_tool.py --make-testset v1_test --data data/datasetv1.npz --split test
    -> data/testsets/v1_test/{testset.npz, manifest.json, videos.csv}

2) Chạy so sánh (mặc định: bộ test v1_test, B0 = models/b0_rule_v1.json, B1 = models/b1_lstm.pt,
   B2 = models/shiftgcn_best.pt):
    python test_tool.py
    python test_tool.py --testset data/testsets/v1_test --b1 models/b1/bi_s42.pt --extra stgcn=models/b2_stgcn.pt
    python test_tool.py --data data/datasetv1.npz --split val          # chạy trên một split của dataset
    python test_tool.py --b0 - --b2 -                                   # chỉ B1 ("-" = bỏ model đó)

Kết quả: output/test_runs/<thời gian>/
    predictions.csv        mỗi cửa sổ một dòng: video, nhãn thật, P(ngã) + dự đoán của từng model
    video_predictions.csv  mỗi video một dòng: nhãn thật, P(ngã) lớn nhất + dự đoán video của từng model
    comparison.csv         mỗi model một dòng: P/R/F1/Acc/Spec/AUC (cửa sổ + video), thời gian xử lý
    errors.csv             mỗi lỗi (model × video) một dòng: FP/FN, hành động, hướng, P(ngã), tốc độ
    by_activity.csv        số video sai theo hành động (ACT) và hướng camera cho từng model
    summary.json           toàn bộ số liệu (để vẽ / ghép báo cáo)
    report.md              báo cáo đọc được ngay (bảng so sánh + phân tích lỗi)

Quy ước: nhãn 0 = ADL (không ngã), 1 = Fall (ngã). Mức VIDEO: video là ngã nếu có >= 1 cửa sổ ngã;
dự đoán video là ngã nếu có >= 1 cửa sổ được dự đoán ngã (giống cảnh báo thật).
Thời gian: đo một cửa sổ (batch 1) sau warm-up trên --device (mặc định cpu, gần thiết bị biên),
cùng quy ước cho mọi model; B0 tính cả trích đặc trưng.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Terminal Windows cũ báo cp1252: không để việc in tiếng Việt làm hỏng một lượt test đã chạy xong.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.b0_rule import B0Rule  # noqa: E402
from src.testset import export_testset, is_testset, load_testset, parse_video_id  # noqa: E402
from src.train_utils import load_splits, metrics, roc_auc  # noqa: E402

DEFAULTS = dict(testset="data/testsets/v1_test", b0="models/b0_rule_v1.json",
                b1="models/b1_lstm.pt", b2="models/shiftgcn_best.pt")


def _path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _rel(path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------- dữ liệu
def load_data(testset=None, data=None, split="test") -> dict:
    """Bộ test chung (ưu tiên) hoặc một split của dataset huấn luyện -> dict thống nhất."""
    if testset:
        ts = load_testset(_path(testset))
        man = ts["manifest"]
        return dict(X=ts["X"], y=ts["y"], video=ts["video"], source=_rel(ts["path"]), kind="testset",
                    name=man.get("name", Path(ts["path"]).name), split=man.get("source", {}).get("split", ""),
                    fps=float(man.get("fps", 10.0)), sha256=man.get("content_sha256"))
    d = load_splits(_path(data))
    X, y, video = d[split]
    from src.testset import content_hash
    return dict(X=X, y=y, video=np.asarray(video).astype(str), source=_rel(_path(data)), kind="dataset",
                name=f"{Path(data).stem}:{split}", split=split, fps=float(d["fps"]),
                sha256=content_hash(X, y, np.asarray(video).astype(str)))


# ---------------------------------------------------------------------------- model
class B0Adapter:
    """Luật B0: nhãn do luật quyết định; score() chỉ để xếp hạng (AUC) và xem độ mạnh cảnh báo."""
    arch = "rule"

    def __init__(self, path):
        payload = json.loads(_path(path).read_text(encoding="utf-8"))
        self.model = B0Rule.from_params(payload.get("params", payload))
        self.threshold = None
        self.num_params = len(self.model.get_params())
        self.size_mb = round(_path(path).stat().st_size / 1e6, 4)
        self.window_size = None
        self.info = dict(params=self.model.get_params())

    def run(self, X):
        return self.model.predict(X).astype(np.int64), self.model.score(X).astype(np.float64)

    def predict_one(self, X1):
        self.model.predict(X1)


class NeuralAdapter:
    """Mọi checkpoint FallClassifier đọc được (lstm, stgcn, shiftgcn, ...)."""

    def __init__(self, path, device):
        from src.classifier import FallClassifier
        self.clf = FallClassifier(_path(path), device=device, verbose=False)
        self.arch = self.clf.arch
        self.threshold = float(self.clf.fall_threshold)
        self.num_params = int(self.clf.num_params)
        self.size_mb = round(_path(path).stat().st_size / 1e6, 3)
        self.window_size = self.clf.window_size
        self.info = dict(model_kwargs=self.clf.meta.get("model_kwargs"), epoch=self.clf.meta.get("epoch")
                         or self.clf.meta.get("best_epoch"), device=str(self.clf.device))

    def run(self, X):
        prob = np.concatenate([self.clf.predict_batch(X[i:i + 512]) for i in range(0, len(X), 512)])
        prob = prob.astype(np.float64)
        if self.clf.gate.enabled:                     # B1 có cổng tốc độ lưu trong checkpoint: tính đúng như App
            return self.clf.gate_labels(X, prob)[0], prob
        return (prob >= self.threshold).astype(np.int64), prob

    def predict_one(self, X1):
        prob = self.clf.predict_batch(X1)
        if self.clf.gate.enabled:
            self.clf.gate_labels(X1, prob)


def load_model(role, path, device):
    p = _path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{role}: không tìm thấy {p}")
    return B0Adapter(p) if p.suffix.lower() == ".json" else NeuralAdapter(p, device)


def time_model(model, X, runs=100, warmup=10) -> dict:
    """Thời gian xử lý MỘT cửa sổ (batch 1): trung bình, trung vị, p95 (ms)."""
    if runs <= 0 or not len(X):
        return dict(mean_ms=None, p50_ms=None, p95_ms=None)
    one = X[:1]
    for _ in range(warmup):
        model.predict_one(one)
    ts = []
    for i in range(runs):
        x = X[i % len(X)][None]
        t0 = time.perf_counter()
        model.predict_one(x)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts)
    return dict(mean_ms=round(float(ts.mean()), 4), p50_ms=round(float(np.median(ts)), 4),
                p95_ms=round(float(np.percentile(ts, 95)), 4))


# ---------------------------------------------------------------------------- chỉ số
def _round(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def video_table(y, video, preds: dict) -> list[dict]:
    """Mỗi video: nhãn thật + với từng model: P(ngã) lớn nhất, số cửa sổ báo ngã, dự đoán video."""
    rows = []
    for v in sorted(set(video)):
        m = video == v
        info = parse_video_id(v)
        row = dict(video_id=str(v), subject=info["subject"], kind=info["kind"], loc=info["loc"], act=info["act"],
                   view=info["view"], n_windows=int(m.sum()), actual_label=int(y[m].max()))
        for key, p in preds.items():
            row[f"{key}_max_prob"] = round(float(p["prob"][m].max()), 4)
            row[f"{key}_fall_windows"] = int(p["pred"][m].sum())
            row[f"{key}_prediction"] = int(p["pred"][m].max())
        rows.append(row)
    return rows


def motion_by_video(X, video, fps) -> dict:
    """Tốc độ vai hạ / thân xoay / góc thân cuối (src/motion.py) lớn nhất trong mỗi video — để giải thích lỗi."""
    from src.motion import motion_features
    f = motion_features(X, fps)
    out = {}
    for v in sorted(set(video)):
        m = video == v
        out[v] = dict(shoulder_drop=round(float(f["shoulder_drop"][m].max()), 2),
                      rotation=round(float(f["rotation"][m].max()), 1),
                      angle_end=round(float(f["angle_end"][m].max()), 1),
                      mask_ratio=round(float(X[m][..., 2].mean()), 3))
    return out


def run(testset=None, data=None, split="test", models: dict | None = None, device="cpu",
        latency_runs=100) -> dict:
    """models = {"B0": "models/b0_rule_v1.json", "B1": "...pt", "B2": "...pt", "<tên khác>": "...pt"}."""
    ds = load_data(testset, data, split)
    X, y, video = ds["X"], ds["y"], ds["video"]
    if not len(y):
        raise ValueError("Bộ test rỗng.")
    if not models:
        raise ValueError("Cần chọn ít nhất một model (--b0, --b1, --b2 hoặc --extra).")

    preds, results = {}, []
    for role, path in models.items():
        model = load_model(role, path, device)
        if model.window_size is not None and model.window_size != X.shape[1]:
            raise ValueError(f"{role} ({_rel(_path(path))}): checkpoint cần cửa sổ {model.window_size} khung, "
                             f"bộ test có {X.shape[1]} khung. Huấn luyện lại model trên dataset cùng cửa sổ.")
        t0 = time.perf_counter()
        pred, prob = model.run(X)
        total_s = time.perf_counter() - t0
        key = role.lower()
        preds[key] = dict(pred=pred, prob=prob)
        vt = video_table(y, video, {key: preds[key]})
        vm = metrics([r["actual_label"] for r in vt], [r[f"{key}_prediction"] for r in vt])
        vm["videos"] = len(vt)
        results.append(dict(
            role=role, key=key, source=_rel(_path(path)), arch=model.arch, threshold=model.threshold,
            params=model.num_params, size_mb=model.size_mb, info=model.info,
            window=_round(metrics(y, pred)), video=_round(vm),
            auc=None if roc_auc(y, prob) is None else round(roc_auc(y, prob), 4),
            latency=time_model(model, X, latency_runs), batch_total_s=round(total_s, 4),
            batch_ms_per_window=round(total_s * 1000.0 / len(y), 4),
        ))

    # ---- dự đoán theo cửa sổ
    window_rows = []
    for i, (lab, vid) in enumerate(zip(y, video)):
        row = dict(window_index=i, video_id=str(vid), actual_label=int(lab), actual_name="fall" if lab else "adl")
        for key, p in preds.items():
            row[f"{key}_probability"] = round(float(p["prob"][i]), 6)
            row[f"{key}_prediction"] = int(p["pred"][i])
            row[f"{key}_correct"] = int(p["pred"][i] == lab)
        window_rows.append(row)

    # ---- theo video + phân tích lỗi
    video_rows = video_table(y, video, preds)
    motion = motion_by_video(X, video, ds["fps"])
    keys = list(preds)
    errors = []
    for r in video_rows:
        wrong_by = [k for k in keys if r[f"{k}_prediction"] != r["actual_label"]]
        r["wrong_by"] = "+".join(k.upper() for k in wrong_by)
        for k in wrong_by:
            errors.append(dict(model=k.upper(), error="FN (bỏ sót)" if r["actual_label"] else "FP (báo nhầm)",
                               video_id=r["video_id"], subject=r["subject"], act=r["act"], view=r["view"],
                               loc=r["loc"], max_prob=r[f"{k}_max_prob"], fall_windows=r[f"{k}_fall_windows"],
                               n_windows=r["n_windows"], also_wrong=r["wrong_by"], **motion[r["video_id"]]))

    by_act = []
    for field in ("act", "view"):
        groups = defaultdict(list)
        for r in video_rows:
            groups[(r["kind"], r[field])].append(r)
        for (kind, val), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], _natural(kv[0][1]))):
            row = dict(group=field, kind=kind, value=val or "-", videos=len(rs))
            for k in keys:
                row[f"{k}_wrong"] = sum(r[f"{k}_prediction"] != r["actual_label"] for r in rs)
            by_act.append(row)

    hard = [r["video_id"] for r in video_rows if keys and all(r[f"{k}_prediction"] != r["actual_label"] for k in keys)]
    report = dict(
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        dataset=dict(name=ds["name"], kind=ds["kind"], source=ds["source"], split=ds["split"],
                     content_sha256=ds["sha256"], windows=int(len(y)), fall_windows=int(y.sum()),
                     adl_windows=int((y == 0).sum()), videos=len(video_rows),
                     fall_videos=sum(r["actual_label"] for r in video_rows), window_size=int(X.shape[1]),
                     fps=ds["fps"]),
        device=device, latency_runs=latency_runs, models=results, hard_videos=hard,
    )
    return dict(report=report, windows=window_rows, videos=video_rows, errors=errors, by_activity=by_act)


def _natural(s):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


# ---------------------------------------------------------------------------- lưu
def _write_csv(path, rows, fields=None):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields or (list(rows[0]) if rows else ["(trống)"]))
        w.writeheader()
        w.writerows(rows)


def comparison_rows(report) -> list[dict]:
    rows = []
    for m in report["models"]:
        w, v, lt = m["window"], m["video"], m["latency"]
        rows.append(dict(
            model=m["role"], arch=m["arch"], checkpoint=m["source"], threshold=m["threshold"],
            precision=w["prec"], recall=w["rec"], f1=w["f1"], accuracy=w["acc"], specificity=w["spec"],
            auc=m["auc"], tp=w["tp"], fp=w["fp"], fn=w["fn"], tn=w["tn"],
            video_precision=v["prec"], video_recall=v["rec"], video_f1=v["f1"], video_specificity=v["spec"],
            video_fn=v["fn"], video_fp=v["fp"], videos=v["videos"],
            latency_ms_mean=lt["mean_ms"], latency_ms_p95=lt["p95_ms"],
            batch_ms_per_window=m["batch_ms_per_window"], params=m["params"], size_mb=m["size_mb"],
        ))
    return rows


def render_markdown(res) -> str:
    rep = res["report"]
    ds = rep["dataset"]
    keys = [m["key"] for m in rep["models"]]
    L = [f"# Kết quả Test Tool — {ds['name']}", "",
         f"- Thời gian chạy: {rep['created_at']} · thiết bị đo: `{rep['device']}`",
         f"- Bộ test: `{ds['source']}` (sha256 `{(ds['content_sha256'] or '')[:12]}…`), "
         f"cửa sổ {ds['window_size']} khung @ {ds['fps']:g} fps",
         f"- {ds['windows']} cửa sổ ({ds['fall_windows']} ngã / {ds['adl_windows']} không ngã), "
         f"{ds['videos']} video ({ds['fall_videos']} ngã / {ds['videos'] - ds['fall_videos']} không ngã)", "",
         "## So sánh theo cửa sổ", "",
         "| Model | Kiến trúc | Ngưỡng | Precision | Recall | F1 | Accuracy | Specificity | AUC | TP | FP | FN | TN |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in rep["models"]:
        w = m["window"]
        thr = "luật" if m["threshold"] is None else f"{m['threshold']:.2f}"
        L.append(f"| {m['role']} | {m['arch']} | {thr} | {w['prec']:.4f} | {w['rec']:.4f} | **{w['f1']:.4f}** | "
                 f"{w['acc']:.4f} | {w['spec']:.4f} | {m['auc'] if m['auc'] is not None else '-'} | "
                 f"{w['tp']} | {w['fp']} | {w['fn']} | {w['tn']} |")
    L += ["", "## So sánh theo video (cảnh báo khi ≥ 1 cửa sổ báo ngã)", "",
          "| Model | Precision | Recall | F1 | Specificity | Bỏ sót (FN) | Báo nhầm (FP) |", "|---|---|---|---|---|---|---|"]
    for m in rep["models"]:
        v = m["video"]
        L.append(f"| {m['role']} | {v['prec']:.4f} | {v['rec']:.4f} | **{v['f1']:.4f}** | {v['spec']:.4f} | "
                 f"{v['fn']} / {v['tp'] + v['fn']} | {v['fp']} / {v['fp'] + v['tn']} |")
    L += ["", f"## Thời gian xử lý (một cửa sổ, batch 1, `{rep['device']}`, {rep['latency_runs']} lần)", "",
          "| Model | Trung bình (ms) | p95 (ms) | Cả bộ test, theo lô (ms/cửa sổ) | Tham số | Dung lượng (MB) |",
          "|---|---|---|---|---|---|"]
    for m in rep["models"]:
        lt = m["latency"]
        L.append(f"| {m['role']} | {lt['mean_ms']} | {lt['p95_ms']} | {m['batch_ms_per_window']} | "
                 f"{m['params']:,} | {m['size_mb']} |")
    L += ["", "## Phân tích lỗi theo video", ""]
    for m in rep["models"]:
        errs = [e for e in res["errors"] if e["model"] == m["role"].upper()]
        fn = [e for e in errs if e["error"].startswith("FN")]
        fp = [e for e in errs if e["error"].startswith("FP")]
        L.append(f"### {m['role']} — bỏ sót {len(fn)}, báo nhầm {len(fp)}")
        if errs:
            L += ["", "| Lỗi | Video | Hành động | Hướng | P(ngã) max (B0: điểm luật) | Cửa sổ báo ngã | Vai hạ (thân/s) | Xoay (°/s) | Góc cuối (°) | Sai cùng |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for e in sorted(errs, key=lambda e: (e["error"], _natural(e["act"]), e["video_id"])):
                L.append(f"| {e['error']} | `{e['video_id']}` | {e['act']} | {e['view']} | {e['max_prob']} | "
                         f"{e['fall_windows']}/{e['n_windows']} | {e['shoulder_drop']} | {e['rotation']} | "
                         f"{e['angle_end']} | {e['also_wrong']} |")
        L.append("")
    acts = [r for r in res["by_activity"] if r["group"] == "act" and any(r[f"{k}_wrong"] for k in keys)]
    if acts:
        L += ["## Hành động có video bị sai", "", "| Loại | Hành động | Số video | " +
              " | ".join(f"{k.upper()} sai" for k in keys) + " |", "|---|---|---|" + "---|" * len(keys)]
        for r in acts:
            L.append(f"| {r['kind']} | {r['value']} | {r['videos']} | " +
                     " | ".join(str(r[f'{k}_wrong']) for k in keys) + " |")
        L.append("")
    views = [r for r in res["by_activity"] if r["group"] == "view"]
    if views:
        L += ["## Theo hướng camera", "", "| Loại | Hướng | Số video | " + " | ".join(f"{k.upper()} sai" for k in keys)
              + " |", "|---|---|---|" + "---|" * len(keys)]
        for r in views:
            L.append(f"| {r['kind']} | {r['value']} | {r['videos']} | " + " | ".join(str(r[f'{k}_wrong']) for k in keys) + " |")
        L.append("")
    L.append(f"**Video mọi model đều sai ({len(rep['hard_videos'])}):** " +
             (", ".join(f"`{v}`" for v in rep["hard_videos"]) or "không có"))
    L += ["", "_Vai hạ / Xoay / Góc cuối: giá trị lớn nhất trong video theo src/motion.py "
          "(ngã thật thường có vai hạ ≥ 3 thân/s hoặc xoay ≥ 100°/s)._", ""]
    return "\n".join(L)


def save(res, output_dir) -> Path:
    out = _path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "predictions.csv", res["windows"])
    _write_csv(out / "video_predictions.csv", res["videos"])
    _write_csv(out / "comparison.csv", comparison_rows(res["report"]))
    _write_csv(out / "errors.csv", res["errors"],
               None if res["errors"] else ["model", "error", "video_id"])
    _write_csv(out / "by_activity.csv", res["by_activity"])
    full = dict(res["report"], errors=res["errors"], by_activity=res["by_activity"])
    (out / "summary.json").write_text(json.dumps(full, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(res), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Test chung B0/B1/B2: nhãn + dự đoán, P/R/F1, thời gian, lỗi.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--testset", default=None, help=f"thư mục bộ test chung (mặc định {DEFAULTS['testset']})")
    ap.add_argument("--data", default=None, help="hoặc: dataset .npz (dùng cùng --split) thay cho --testset")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--b0", default=DEFAULTS["b0"], help="JSON tham số B0; '-' = bỏ")
    ap.add_argument("--b1", default=DEFAULTS["b1"], help="checkpoint B1 (LSTM); '-' = bỏ")
    ap.add_argument("--b2", default=DEFAULTS["b2"], help="checkpoint B2 (Shift-GCN / ST-GCN); '-' = bỏ")
    ap.add_argument("--extra", action="append", default=[], metavar="TÊN=ĐƯỜNG_DẪN",
                    help="thêm model để so sánh, lặp lại được (vd --extra stgcn=models/b2_stgcn.pt)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (đo thời gian nên dùng cpu)")
    ap.add_argument("--latency-runs", type=int, default=100, help="số lần đo một cửa sổ (0 = bỏ)")
    ap.add_argument("--out", default=None, help="thư mục kết quả (mặc định output/test_runs/<thời gian>)")
    ap.add_argument("--make-testset", metavar="TÊN", default=None,
                    help="tạo bộ test chung data/testsets/<TÊN> từ --data/--split rồi thoát")
    ap.add_argument("--note", default="", help="ghi chú lưu vào manifest khi tạo bộ test")
    args = ap.parse_args(argv)

    if args.make_testset:
        if not args.data:
            ap.error("--make-testset cần --data (vd data/datasetv1.npz)")
        out = export_testset(_path(args.data), _path(f"data/testsets/{args.make_testset}"), args.split,
                             name=args.make_testset, note=args.note)
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        c = man["counts"]
        print(f"Đã tạo bộ test {out}: {c['windows']} cửa sổ ({c['fall_windows']} ngã), {c['videos']} video "
              f"({c['fall_videos']} ngã), subject {man['subjects']}, sha256 {man['content_sha256'][:12]}…")
        if man["label_kind_mismatch"]:
            print(f"[cảnh báo] video có nhãn khác tên loại: {man['label_kind_mismatch']}")
        return 0

    testset = args.testset
    if not testset and not args.data:
        testset = DEFAULTS["testset"]
        if not is_testset(_path(testset)):
            ap.error(f"Chưa có bộ test {testset}. Tạo: python test_tool.py --make-testset v1_test "
                     f"--data data/datasetv1.npz --split test")
    models = {}
    for role in ("b0", "b1", "b2"):
        val = getattr(args, role)
        if val and val != "-":
            models[role.upper()] = val
    for item in args.extra:
        if "=" not in item:
            ap.error(f"--extra cần dạng TÊN=ĐƯỜNG_DẪN, nhận {item}")
        name, path = item.split("=", 1)
        models[name.strip()] = path.strip()
    try:
        res = run(testset, args.data, args.split, models, args.device, args.latency_runs)
        out = save(res, args.out or f"output/test_runs/{datetime.now():%Y%m%d_%H%M%S}")
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))

    rep = res["report"]
    ds = rep["dataset"]
    print(f"\nBộ test {ds['name']}: {ds['windows']} cửa sổ, {ds['videos']} video ({ds['fall_videos']} ngã)\n")
    print(f"{'model':8s}{'arch':>9s}{'P':>8s}{'R':>8s}{'F1':>8s}{'AUC':>8s}{'F1 video':>10s}"
          f"{'FN vid':>8s}{'FP vid':>8s}{'ms/cửa sổ':>11s}")
    for m in rep["models"]:
        w, v = m["window"], m["video"]
        print(f"{m['role']:8s}{m['arch']:>9s}{w['prec']:8.4f}{w['rec']:8.4f}{w['f1']:8.4f}"
              f"{(m['auc'] or 0):8.4f}{v['f1']:10.4f}{v['fn']:8d}{v['fp']:8d}{m['latency']['mean_ms'] or 0:11.3f}")
    if rep["hard_videos"]:
        print(f"\nVideo mọi model đều sai: {rep['hard_videos']}")
    print(f"\nKết quả: {_rel(out)}/  (report.md, comparison.csv, predictions.csv, video_predictions.csv, errors.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
