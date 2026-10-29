#!/usr/bin/env python3
"""
check_labels.py
---------------
Kiểm tra (và tùy chọn sửa) nhãn của dataset cửa sổ — nguyên nhân chính khiến model báo "NGÃ" khi người chỉ
NẰM bình thường: nhiều cửa sổ có nhãn ngã thực chất là người đang nằm yên (sau cú ngã, hoặc cả video ngã bị
gán nhãn 1 do thiếu khoảng ngã). Model học được "nằm ngang = ngã" thay vì "chuyển động rơi = ngã".

Báo cáo:
  1. Video ngã mà TOÀN BỘ cửa sổ là nhãn ngã (thường do thiếu dòng trong fall_intervals.csv:
     build_dataset.py khi đó gán nhãn 1 cho cả video, gồm cả lúc đứng và lúc nằm).
  2. Video ngã không có cửa sổ ngã nào (khoảng ngã quá ngắn / lệch thời gian).
  3. Cửa sổ "NẰM YÊN" mang nhãn ngã: thân nằm ngang gần suốt 2 s, hông gần như không đổi độ cao, ít chuyển động.
     Các cửa sổ này trông y hệt người nằm nghỉ, là "dạy" model rằng nằm = ngã.

Sửa (--fix): đổi nhãn các cửa sổ ở mục 3 thành 0 (không ngã), ghi ra file mới. Cú ngã vẫn được học từ
các cửa sổ có chuyển động rơi. File gốc không bị sửa.

  python check_labels.py data/dataset.npz
  python check_labels.py data/dataset.npz --fix --out data/dataset_clean.npz
  python train.py --arch lstm --data data/dataset_clean.npz --out models/lstm_clean.pt

Lưu ý: chỉ sửa train và val. Tập TEST giữ nguyên nhãn gốc để so sánh công bằng với model cũ
(thêm --fix-test nếu nhóm thống nhất định nghĩa "nằm yên không phải ngã" cho cả đánh giá).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SHOULDERS, HIPS = [5, 6], [11, 12]


def window_features(X: np.ndarray) -> dict:
    """X [N,T,17,3] đã chuẩn hóa (kênh 2 = mask) -> đặc trưng tư thế/chuyển động mỗi cửa sổ."""
    m = X[..., 2] > 0
    sh, hp = X[:, :, SHOULDERS, :2].mean(2), X[:, :, HIPS, :2].mean(2)
    ok = m[:, :, SHOULDERS].all(2) & m[:, :, HIPS].all(2)
    torso = sh - hp
    horiz = (np.abs(torso[..., 0]) > np.abs(torso[..., 1])) & ok        # thân nằm ngang
    seen = ok.sum(1)
    horiz_ratio = np.where(seen > 0, horiz.sum(1) / np.maximum(seen, 1), 0.0)
    hy = np.where(ok, hp[..., 1], np.nan)
    with warnings.catch_warnings():                                       # cửa sổ không thấy hông -> NaN -> 0
        warnings.simplefilter("ignore", RuntimeWarning)
        hip_range = np.nan_to_num(np.nanmax(hy, 1) - np.nanmin(hy, 1), nan=0.0)   # đơn vị: độ dài xương
    both = m[:, 1:] & m[:, :-1]
    step = np.linalg.norm(np.diff(X[..., :2], axis=1), axis=-1) * both
    motion = step.sum((1, 2)) / np.maximum(both.sum((1, 2)), 1)               # dịch chuyển TB / khớp / khung
    return dict(horiz=horiz_ratio, hip_range=hip_range, motion=motion, seen=seen)


def static_lying(f: dict, horiz=0.8, hip_range=0.5, motion=0.08) -> np.ndarray:
    return (f["horiz"] >= horiz) & (f["hip_range"] < hip_range) & (f["motion"] < motion)


def load_raw(path: Path) -> tuple[dict, str]:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    fmt = "code_1" if "X_train" in d else ("web" if "X" in d else None)
    if fmt is None:
        sys.exit(f"{path}: không nhận ra định dạng dataset")
    return d, fmt


def split_views(d: dict, fmt: str):
    """-> list (tên tập, X, y, video, hàm ghi nhãn mới)."""
    out = []
    if fmt == "code_1":
        for s in ("train", "val", "test"):
            out.append((s, d[f"X_{s}"], d[f"y_{s}"], d[f"video_{s}"].astype(str), (s, None)))
    else:
        split = d["split"].astype(str) if "split" in d else np.array([""] * len(d["y"]))
        for s in sorted(set(split)):
            m = split == s
            out.append((s or "(trống)", d["X"][m], d["y"][m], d["video"][m].astype(str), (None, m)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kiểm tra / sửa nhãn 'nằm yên' bị gán là ngã",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("data", nargs="?", default="data/dataset.npz")
    ap.add_argument("--fix", action="store_true", help="đổi nhãn cửa sổ nằm yên (đang là ngã) thành 0")
    ap.add_argument("--fix-test", action="store_true", help="sửa cả tập test (mặc định giữ nguyên)")
    ap.add_argument("--out", default=None, help="file ra khi --fix (mặc định <tên>_clean.npz)")
    ap.add_argument("--horiz", type=float, default=0.8, help="tỉ lệ khung thân nằm ngang tối thiểu")
    ap.add_argument("--hip-range", type=float, default=0.5, help="hông lên/xuống tối đa (độ dài xương)")
    ap.add_argument("--motion", type=float, default=0.08, help="chuyển động TB tối đa")
    ap.add_argument("--list", type=int, default=15, help="số video liệt kê mỗi mục")
    args = ap.parse_args(argv)

    path = Path(args.data)
    if not path.exists() and (ROOT / path).exists():
        path = ROOT / path
    d, fmt = load_raw(path)
    print(f"Dataset: {path} (định dạng {fmt})\n")
    changes = {}
    for name, X, y, vid, key in split_views(d, fmt):
        f = window_features(X)
        lying = static_lying(f, args.horiz, args.hip_range, args.motion)
        adl = np.array(["fall" not in v.lower().split("_")[1:-1] and "fall" not in v.lower() for v in vid])
        fall_v = sorted({v for v, a in zip(vid, adl) if not a})
        whole = [v for v in fall_v if y[vid == v].all() and (vid == v).sum() >= 4]
        none = [v for v in fall_v if not y[vid == v].any()]
        bad = lying & (y == 1)
        print(f"[{name}] {len(y)} cửa sổ, {int(y.sum())} nhãn ngã, {len(fall_v)} video ngã")
        print(f"  1. Video ngã có TOÀN BỘ cửa sổ là ngã (>= 4 cửa sổ, nghi thiếu khoảng ngã): {len(whole)}")
        for v in whole[:args.list]:
            m = vid == v
            pose = "".join("N" if h >= args.horiz else ("n" if h >= 0.3 else ".") for h in f["horiz"][m])
            print(f"       {v:22s} {int(m.sum()):3d} cửa sổ | tư thế {pose}  (N = nằm, . = đứng/ngồi)")
        print(f"  2. Video ngã KHÔNG có cửa sổ ngã: {len(none)}" + (f"  {', '.join(none[:args.list])}" if none else ""))
        print(f"  3. Cửa sổ NẰM YÊN nhưng nhãn NGÃ: {int(bad.sum())}/{int(y.sum())} ({bad.sum() / max(1, y.sum()):.0%})"
              f" | cửa sổ nằm yên nhãn 0 (sinh hoạt / sau ngã): {int((lying & (y == 0)).sum())}")
        per_video = sorted({v: int((bad & (vid == v)).sum()) for v in set(vid[bad])}.items(), key=lambda kv: -kv[1])
        if per_video:
            print("       nhiều nhất: " + ", ".join(f"{v} ({n})" for v, n in per_video[:8]))
        print()
        if args.fix and (name != "test" or args.fix_test) and bad.any():
            changes[key] = bad

    if not args.fix:
        print("Chạy thêm --fix để đổi nhãn mục 3 thành 0 (ghi file mới, không sửa file gốc).")
        return 0
    new = {k: v.copy() for k, v in d.items()}
    total = 0
    for (s, mask), bad in changes.items():
        if fmt == "code_1":
            new[f"y_{s}"][bad] = 0
        else:
            idx = np.flatnonzero(mask)[bad]
            new["y"][idx] = 0
        total += int(bad.sum())
    out = Path(args.out) if args.out else path.with_name(path.stem + "_clean.npz")
    np.savez_compressed(out, **new)
    print(f"Đã đổi {total} nhãn ngã -> 0 (nằm yên) {'gồm cả test' if args.fix_test else '(train/val; test giữ nguyên)'}.")
    print(f"Đã ghi {out}\nTrain lại: python train.py --arch lstm --data {out.as_posix()} --out models/lstm_clean.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
