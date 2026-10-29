"""
train_utils.py
--------------
Phần KHÔNG cần PyTorch của việc huấn luyện / đánh giá (dùng chung cho train.py và evaluate.py):
đọc dataset, tăng cường dữ liệu, chỉ số đánh giá, chọn ngưỡng.

Đọc được HAI định dạng dataset:
  1. code_1  (build_dataset.py):  X_train, y_train, video_train, X_val, ..., X_test, ...
  2. web     (run_web.py build-dataset / nút "Tạo dataset"):  X, y, video, subject, split, fps, window_size
     -> chia theo cột split; nếu split trống thì tự chia THEO SUBJECT (giống build_dataset.py).

Quy ước X: [N, T, 17, 3], kênh 0-1 = tọa độ đã chuẩn hóa (features.normalize_clip), kênh 2 = mask.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import FLIP_INDEX, NUM_KEYPOINTS

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------- đọc dataset
def subject_of(video_id: str) -> str:
    """'Subject 1_Fall_01' -> 'Subject 1' (quy ước video_id của extract_skeleton.py)."""
    return str(video_id).split("_")[0]


def split_subjects(subjects, val_frac=0.15, test_frac=0.15, seed=0):
    """Chia subject thành train/val/test — giống hệt build_dataset.py của code_1."""
    subjects = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(subjects)))
    n = len(subjects)
    n_test = max(1, round(n * test_frac)) if n >= 3 else 0
    n_val = max(1, round(n * val_frac)) if n >= 3 else 0
    test = {subjects[i] for i in order[:n_test]}
    val = {subjects[i] for i in order[n_test:n_test + n_val]}
    train = set(subjects) - test - val
    return train, val, test


def check_split(X, y, name):
    """Kiểm tra giống train.py của code_1."""
    if len(X) != len(y):
        raise ValueError(f"{name}: số lượng X và y không khớp.")
    if len(X) == 0:
        raise ValueError(f"{name}: tập dữ liệu rỗng.")
    if X.ndim != 4 or X.shape[2:] != (NUM_KEYPOINTS, 3):
        raise ValueError(f"{name}: cần X dạng [N,T,17,3], nhận {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: X chứa NaN hoặc Inf.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name}: nhãn chỉ được là 0 hoặc 1.")


def load_splits(path, fps: float | None = None, seed: int = 0) -> dict:
    """Đọc dataset -> {"train": (X, y, video), "val": ..., "test": ..., "fps": .., "window_size": .., "format": ..}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}")
    with np.load(path, allow_pickle=False) as d:
        files = set(d.files)
        out: dict = {}
        if {"X_train", "y_train"} <= files:                      # định dạng code_1
            for s in SPLITS:
                X = d[f"X_{s}"].astype(np.float32)
                y = d[f"y_{s}"].astype(np.int64)
                vid = d[f"video_{s}"] if f"video_{s}" in files else np.array([""] * len(y))
                out[s] = (X, y, vid.astype(str))
            out["format"] = "code_1"
            out["fps"] = float(fps or 10.0)                      # code_1 không lưu fps (quy ước 10)
        elif {"X", "y"} <= files:                                 # định dạng web
            X, y = d["X"].astype(np.float32), d["y"].astype(np.int64)
            vid = d["video"].astype(str) if "video" in files else np.array([""] * len(y))
            split = d["split"].astype(str) if "split" in files else np.array([""] * len(y))
            subj = d["subject"].astype(str) if "subject" in files else np.array([""] * len(y))
            subj = np.where(subj == "", [subject_of(v) for v in vid], subj)
            if not all((split == s).any() for s in SPLITS):
                # metadata chưa điền đủ split -> tự chia theo subject như build_dataset.py
                tr, va, te = split_subjects(subj, seed=seed)
                split = np.where(np.isin(subj, list(te)), "test",
                                 np.where(np.isin(subj, list(va)), "val", "train"))
                out["auto_split"] = {"train": sorted(tr), "val": sorted(va), "test": sorted(te)}
            for s in SPLITS:
                m = split == s
                out[s] = (X[m], y[m], vid[m])
            out["format"] = "web"
            out["fps"] = float(fps or (float(d["fps"]) if "fps" in files else 10.0))
        else:
            raise ValueError(f"{path}: không nhận ra định dạng dataset (khóa: {sorted(files)})")
    out["window_size"] = int(out["train"][0].shape[1])
    return out


def describe(splits: dict) -> str:
    lines = []
    for s in SPLITS:
        X, y, vid = splits[s]
        subj = sorted({subject_of(v) for v in vid if v})
        lines.append(f"{s:5s}: {len(y):5d} cửa sổ | ngã {int(y.sum()):4d} | không ngã {int((y == 0).sum()):4d}"
                     f" | {len(set(vid))} video | {', '.join(subj) if subj else '-'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- tăng cường dữ liệu
def flip_batch(X):
    """Lật ngang: hoán đổi trái/phải và đổi dấu x (x đã quy về tâm hông nên có thể âm). Ô thiếu giữ 0."""
    out = X[:, :, FLIP_INDEX, :].copy()
    observed = out[..., 2] > 0
    x = out[..., 0]
    x[observed] *= -1.0
    x[~observed] = 0.0
    return out


def jitter_batch(X, std=0.01, rng=None):
    """Nhiễu Gauss nhẹ trên tọa độ của khớp quan sát được; ô thiếu giữ 0."""
    rng = rng or np.random.default_rng()
    out = X.copy()
    noise = rng.normal(0.0, std, size=out[..., :2].shape).astype(np.float32)
    observed = out[..., 2:3] > 0
    out[..., :2] += noise * observed
    out[..., :2][out[..., 2] <= 0] = 0.0
    return out


def augment(xb, rng, flip_p=0.5, jitter_std=0.01):
    xb = xb.copy()
    if flip_p > 0:
        m = rng.random(len(xb)) < flip_p
        if m.any():
            xb[m] = flip_batch(xb[m])
    if jitter_std > 0:
        xb = jitter_batch(xb, std=jitter_std, rng=rng)
    return xb


# ---------------------------------------------------------------------------- đánh giá
def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, spec=spec, tp=tp, fp=fp, fn=fn, tn=tn)


def metrics_at(y_true, p_fall, threshold=0.5) -> dict:
    m = metrics(y_true, (np.asarray(p_fall) >= threshold).astype(np.int64))
    m["threshold"] = float(threshold)
    return m


def best_threshold(y_true, p_fall, grid=None) -> float:
    """Ngưỡng cho F1 lớp ngã cao nhất (chọn trên VAL, không bao giờ trên test).
    Hòa thì lấy ngưỡng gần 0.5 nhất."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    best, best_key = 0.5, (-1.0, 0.0)
    for t in grid:
        key = (metrics_at(y_true, p_fall, t)["f1"], -abs(t - 0.5))
        if key > best_key:
            best, best_key = float(t), key
    return best


def roc_auc(y_true, p_fall) -> float | None:
    """AUC ROC không cần sklearn (thống kê Mann-Whitney). None nếu chỉ có một lớp."""
    y = np.asarray(y_true)
    p = np.asarray(p_fall, dtype=np.float64)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):                                   # hạng trung bình cho giá trị bằng nhau
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def video_level(y_true, p_fall, videos, threshold=0.5) -> dict:
    """Chỉ số theo VIDEO: video có ít nhất một cửa sổ ngã thật là video ngã;
    dự đoán ngã nếu có ít nhất một cửa sổ P(ngã) >= ngưỡng (giống cảnh báo thật)."""
    videos = np.asarray(videos).astype(str)
    yt, yp = [], []
    for v in sorted(set(videos)):
        m = videos == v
        yt.append(int(np.asarray(y_true)[m].max()))
        yp.append(int((np.asarray(p_fall)[m] >= threshold).any()))
    out = metrics(yt, yp)
    out["videos"] = len(yt)
    return out


def format_metrics(m: dict) -> str:
    return (f"Acc {m['acc']:.4f} | P {m['prec']:.4f} | R {m['rec']:.4f} | F1 {m['f1']:.4f} | "
            f"Spec {m['spec']:.4f} | TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']}")


# ---------------------------------------------------------------------------- so sánh model
def roc_curve(y_true, p_fall, max_points=200) -> list[list[float]]:
    """Điểm [FPR, TPR] của đường ROC (giảm mẫu còn <= max_points), bắt đầu (0,0), kết thúc (1,1)."""
    y = np.asarray(y_true)
    p = np.asarray(p_fall, dtype=np.float64)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return [[0.0, 0.0], [1.0, 1.0]]
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    distinct = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]     # điểm cuối mỗi nhóm giá trị bằng nhau
    tpr = np.cumsum(ys == 1)[distinct] / n1
    fpr = np.cumsum(ys == 0)[distinct] / n0
    pts = np.c_[np.r_[0.0, fpr], np.r_[0.0, tpr]]
    if len(pts) > max_points:
        keep = np.unique(np.r_[np.linspace(0, len(pts) - 1, max_points).round().astype(int), len(pts) - 1])
        pts = pts[keep]
    return [[round(float(a), 4), round(float(b), 4)] for a, b in pts]


def f1_curve(y_true, p_fall, grid=None) -> list[list[float]]:
    """[ngưỡng, F1] để vẽ F1 theo ngưỡng."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    return [[float(t), round(metrics_at(y_true, p_fall, t)["f1"], 4)] for t in grid]


def _f1_from_counts(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def bootstrap_by_video(y_true, preds: dict, videos, n_boot=1000, seed=0) -> dict:
    """Khoảng tin cậy 95% của F1 (theo cửa sổ) bằng bootstrap THEO VIDEO (lấy mẫu lại cả video, vì các
    cửa sổ cùng video chồng nhau, không độc lập). preds = {tên model: dự đoán 0/1}.
    Trả {"ci": {tên: [thấp, cao]}, "diff": {"a|b": [thấp, cao, tỉ lệ b > a]}}."""
    y = np.asarray(y_true)
    videos = np.asarray(videos).astype(str)
    uniq = np.unique(videos)
    idx_of = [np.flatnonzero(videos == v) for v in uniq]
    names = list(preds)
    # đếm tp, fp, fn theo từng video rồi cộng -> nhanh hơn ghép chỉ số
    counts = {}
    for name in names:
        pr = np.asarray(preds[name])
        counts[name] = np.array([[int(((pr[i] == 1) & (y[i] == 1)).sum()), int(((pr[i] == 1) & (y[i] == 0)).sum()),
                                  int(((pr[i] == 0) & (y[i] == 1)).sum())] for i in idx_of])
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(n_boot) for name in names}
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        for name in names:
            tp, fp, fn = counts[name][pick].sum(axis=0)
            samples[name][b] = _f1_from_counts(tp, fp, fn)
    out = {"ci": {n: [round(float(np.percentile(samples[n], 2.5)), 4),
                      round(float(np.percentile(samples[n], 97.5)), 4)] for n in names}, "diff": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = samples[b] - samples[a]
            out["diff"][f"{a}|{b}"] = [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4),
                                       round(float((d > 0).mean()), 3)]
    return out


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Kiểm định McNemar chính xác (nhị thức) trên các cửa sổ hai model đúng/sai khác nhau.
    Lưu ý: cửa sổ cùng video không độc lập nên p-value chỉ mang tính tham khảo."""
    from math import comb
    y = np.asarray(y_true)
    ca, cb = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    b01, b10 = int((ca & ~cb).sum()), int((~ca & cb).sum())     # chỉ A đúng / chỉ B đúng
    n, k = b01 + b10, min(b01, b10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_a_correct": b01, "only_b_correct": b10, "p_value": round(float(p), 4)}
