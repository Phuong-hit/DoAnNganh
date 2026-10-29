"""So sánh nhiều bộ phân loại ngã (ST-GCN, LSTM, ...) trên CÙNG một tập dữ liệu.

Dùng cho trang /compare của web và `python evaluate.py ... --html`. Không import torch ở đầu tệp:
FallClassifier chỉ được nạp khi chạy so sánh.

Kết quả (dict, chuyển thẳng sang JSON):
  dataset   : đường dẫn, tập (test/val), số cửa sổ, số cửa sổ ngã, số video, người
  models    : mỗi model — chỉ số ở ngưỡng 0.5 và ngưỡng của checkpoint, AUC, ROC, F1 theo ngưỡng,
              ma trận nhầm lẫn, chỉ số theo video, khoảng tin cậy 95% của F1 (bootstrap theo video),
              số tham số, dung lượng, thời gian suy luận một cửa sổ
  pairs     : từng cặp model — khoảng tin cậy chênh lệch F1, McNemar
  videos    : từng video — nhãn thật, P(ngã) lớn nhất và kết luận của từng model
  windows   : từng video — nhãn và P(ngã) của mọi cửa sổ (để vẽ chi tiết)
  verdict   : câu kết luận ngắn
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.train_utils import (bootstrap_by_video, f1_curve, load_splits, mcnemar, metrics_at, roc_auc,
                             roc_curve, subject_of, video_level)

from .config import resolve

METRIC_KEYS = ("acc", "prec", "rec", "f1", "spec")


def list_datasets() -> list[str]:
    """Các file dataset cửa sổ trong project (data/*.npz, output/**/dataset.npz, output/**/windows.npz)."""
    root = resolve(".")
    found = set()
    for pat in ("data/*.npz", "output/dataset.npz", "output/**/dataset/windows.npz", "output/**/dataset.npz"):
        for p in root.glob(pat):
            if p.is_file():
                found.add(p.relative_to(root).as_posix())
    return sorted(found)


def _latency_ms(clf, X, runs: int) -> float | None:
    if runs <= 0 or not len(X):
        return None
    one = X[:1]
    for _ in range(min(10, runs)):
        clf.predict_batch(one)
    t0 = time.perf_counter()
    for _ in range(runs):
        clf.predict_batch(one)
    return round((time.perf_counter() - t0) * 1000.0 / runs, 3)


def _r(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def _status(true_fall: bool, pred_fall: bool) -> str:
    if true_fall:
        return "TP" if pred_fall else "FN"
    return "FP" if pred_fall else "TN"


def compare_models(checkpoints, data_path, split: str = "test", device: str | None = "cpu",
                   runs: int = 100, n_boot: int = 1000, loader=None) -> dict:
    """checkpoints: danh sách đường dẫn .pt. loader(path, device) -> đối tượng giống FallClassifier
    (mặc định FallClassifier; test truyền bộ giả)."""
    if loader is None:
        from src.classifier import FallClassifier

        def loader(path, device):
            return FallClassifier(path, device=device, verbose=False)

    paths = [resolve(p) if not Path(p).is_absolute() else Path(p) for p in checkpoints]
    if not paths:
        raise ValueError("Chọn ít nhất một model.")
    data_file = resolve(data_path) if not Path(data_path).is_absolute() else Path(data_path)
    data = load_splits(data_file)
    if split not in ("train", "val", "test"):
        raise ValueError("split phải là train | val | test")
    X, y, vid = data[split]
    if not len(y):
        raise ValueError(f"Tập {split} rỗng.")
    vid = np.asarray(vid).astype(str)

    models, probs, warn = [], {}, []
    used_names: set[str] = set()
    for p in paths:
        clf = loader(p, device)
        name = p.stem if p.stem not in used_names else f"{p.stem}_{len(used_names)}"
        used_names.add(name)
        if clf.window_size != X.shape[1]:
            warn.append(f"{name}: checkpoint dùng cửa sổ {clf.window_size} khung, dataset {X.shape[1]} khung")
        prob = np.asarray(clf.predict_batch(X), dtype=np.float64)
        thr = float(clf.fall_threshold)
        probs[name] = (prob, thr)
        m05, mthr = metrics_at(y, prob, 0.5), metrics_at(y, prob, thr)
        mv = video_level(y, prob, vid, thr)
        auc = roc_auc(y, prob)
        models.append(dict(
            name=name, file=p.as_posix() if not p.is_relative_to(resolve(".")) else p.relative_to(resolve(".")).as_posix(),
            arch=getattr(clf, "arch", "?"), params=int(getattr(clf, "num_params", 0)),
            size_mb=round(p.stat().st_size / 1e6, 2) if p.exists() else None,
            latency_ms=_latency_ms(clf, X, runs), device=str(getattr(clf, "device", "")),
            threshold=round(thr, 2), auc=None if auc is None else round(auc, 4),
            at05=_r(m05), atthr=_r(mthr), video=_r(mv),
            roc=roc_curve(y, prob), f1_curve=f1_curve(y, prob),
            meta={k: v for k, v in (getattr(clf, "meta", {}) or {}).items()
                  if k in ("epoch", "val_f1", "val_acc", "model_kwargs", "fps", "window_size", "seed")},
        ))

    preds_thr = {n: (pr >= t).astype(int) for n, (pr, t) in probs.items()}
    boot = bootstrap_by_video(y, preds_thr, vid, n_boot=n_boot) if n_boot > 0 else {"ci": {}, "diff": {}}
    for m in models:
        m["f1_ci"] = boot["ci"].get(m["name"])
    names = [m["name"] for m in models]
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = boot["diff"].get(f"{a}|{b}")
            pairs.append(dict(a=a, b=b, f1_diff=round(models[names.index(b)]["atthr"]["f1"]
                                                    - models[names.index(a)]["atthr"]["f1"], 4),
                              f1_diff_ci=d[:2] if d else None, prob_b_better=d[2] if d else None,
                              mcnemar=mcnemar(y, preds_thr[a], preds_thr[b])))

    videos, windows = [], {}
    for v in sorted(set(vid)):
        m = vid == v
        true_fall = bool(y[m].max())
        v = str(v)
        row = dict(video=v, subject=subject_of(v), label="fall" if true_fall else "adl",
                   windows=int(m.sum()), fall_windows=int(y[m].sum()), models={})
        for n, (pr, t) in probs.items():
            mx = float(pr[m].max())
            row["models"][n] = dict(max_p=round(mx, 3), pred=int(mx >= t), status=_status(true_fall, mx >= t),
                                    fall_windows=int((pr[m] >= t).sum()))
        row["disagree"] = len({r["pred"] for r in row["models"].values()}) > 1
        videos.append(row)
        windows[v] = dict(y=y[m].astype(int).tolist(),
                          p={n: [round(float(x), 3) for x in pr[m]] for n, (pr, _) in probs.items()})

    return dict(
        dataset=dict(path=data_file.relative_to(resolve(".")).as_posix() if data_file.is_relative_to(resolve("."))
                     else data_file.as_posix(),
                     format=data["format"], split=split, windows=int(len(y)), fall=int(y.sum()),
                     no_fall=int((y == 0).sum()), videos=len(set(vid)),
                     subjects=sorted({subject_of(v) for v in vid}), window_size=int(X.shape[1]),
                     fps=data["fps"]),
        models=models, pairs=pairs, videos=videos, windows=windows, warnings=warn,
        verdict=verdict(models, pairs), generated=time.strftime("%Y-%m-%d %H:%M"),
    )


def verdict(models: list[dict], pairs: list[dict]) -> list[str]:
    """Vài câu kết luận ngắn, dựa trên F1 ở ngưỡng của từng checkpoint (ngưỡng chọn trên val)."""
    if not models:
        return []
    out = []
    best = max(models, key=lambda m: (m["atthr"]["f1"], m["atthr"]["rec"]))
    out.append(f"F1 cao nhất: {best['name']} ({best['atthr']['f1']:.3f} ở ngưỡng {best['threshold']:.2f}).")
    best_rec = max(models, key=lambda m: (m["atthr"]["rec"], m["atthr"]["f1"]))
    if best_rec is not best:
        out.append(f"Recall (bắt được ngã) cao nhất: {best_rec['name']} ({best_rec['atthr']['rec']:.3f}).")
    for p in pairs:
        ci = p["f1_diff_ci"]
        if ci is None:
            continue
        lead, lag = (p["b"], p["a"]) if p["f1_diff"] > 0 else (p["a"], p["b"])
        gap = abs(p["f1_diff"])
        if gap < 0.0005:
            out.append(f"{p['a']} và {p['b']} có F1 bằng nhau ({models[0]['atthr']['f1']:.3f}).")
        elif ci[0] > 0 or ci[1] < 0:
            out.append(f"{lead} hơn {lag} {gap:.3f} F1; khoảng tin cậy 95% của chênh lệch "
                       f"[{ci[0]:+.3f}, {ci[1]:+.3f}] không chứa 0 → khác biệt đáng tin.")
        else:
            out.append(f"{lead} hơn {lag} {gap:.3f} F1 nhưng khoảng tin cậy 95% [{ci[0]:+.3f}, {ci[1]:+.3f}] "
                       "chứa 0 → chưa đủ bằng chứng là khác nhau (tập test nhỏ); nên so thêm nhiều seed.")
    fast = [m for m in models if m["latency_ms"] is not None]
    if len(fast) > 1:
        f = min(fast, key=lambda m: m["latency_ms"])
        s = max(fast, key=lambda m: m["latency_ms"])
        if f is not s and s["latency_ms"] >= 1.1 * max(f["latency_ms"], 1e-6):
            out.append(f"{f['name']} nhanh hơn {s['latency_ms'] / max(f['latency_ms'], 1e-6):.1f} lần "
                       f"({f['latency_ms']:.2f} so với {s['latency_ms']:.2f} ms/cửa sổ) và có "
                       f"{f['params']:,} so với {s['params']:,} tham số.")
    return out


def to_csv_rows(result: dict) -> list[dict]:
    rows = []
    for m in result["models"]:
        r = dict(model=m["name"], arch=m["arch"], file=m["file"], params=m["params"], size_mb=m["size_mb"],
                 latency_ms=m["latency_ms"], auc=m["auc"], threshold=m["threshold"])
        for k in METRIC_KEYS:
            r[f"{k}@0.5"] = m["at05"][k]
        for k in METRIC_KEYS:
            r[f"{k}@thr"] = m["atthr"][k]
        r.update({f"{k}@0.5": m["at05"][k] for k in ("tp", "fn", "fp", "tn")})
        r["f1_ci95"] = "" if not m["f1_ci"] else f"{m['f1_ci'][0]}–{m['f1_ci'][1]}"
        r.update(video_recall=m["video"]["rec"], video_spec=m["video"]["spec"],
                 video_missed=m["video"]["fn"], video_false_alarm=m["video"]["fp"])
        rows.append(r)
    return rows


def render_report(result: dict | None, standalone: bool = True, options: dict | None = None) -> str:
    """HTML của trang so sánh. standalone=True: một file tự chứa (dữ liệu nhúng sẵn, không cần server)
    để đính kèm báo cáo; False: trang /compare của web (có ô chọn model và nút So sánh)."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
                      autoescape=select_autoescape(["html"]))
    return env.get_template("compare.html").render(data=result, standalone=standalone,
                                                   options=options or {})
