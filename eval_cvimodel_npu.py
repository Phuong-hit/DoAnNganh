"""Đo .cvimodel Shift-GCN bằng mô phỏng NPU của TPU-MLIR, so với PyTorch. Chạy trong WSL (môi trường tpu_mlir),
do maixcam_app/convert_shiftgcn.sh gọi:

  python eval_cvimodel_npu.py eval.npz a_int8.cvimodel b_bf16.cvimodel --out choice.txt

eval.npz: clip [N,3,17,17], p [N] (P(ngã) của PyTorch), threshold.
Chọn bản ĐẦU TIÊN (theo thứ tự truyền vào) không đổi nhãn nào ở ngưỡng và lệch P trung bình < 0.02;
không bản nào đạt thì chọn bản ít đổi nhãn nhất.
"""
import argparse
import time

import numpy as np
import tpu_mlir  # noqa: F401  (thiết lập đường dẫn thư viện của tpu_mlir)
from tpu_mlir.python.tools.model_runner import model_inference


def probs(model, clips, limit_s=900):
    out, t0 = [], time.time()
    for i in range(len(clips)):
        r = model_inference({"clip": clips[i:i + 1]}, model)
        z = np.asarray(list(r.values())[-1], dtype=np.float64).reshape(-1)
        z = z - z.max()
        out.append(float(np.exp(z[1]) / np.exp(z).sum()))
        if time.time() - t0 > limit_s:                      # quá lâu: dừng, đo trên phần đã chạy
            break
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default="choice.txt")
    ap.add_argument("--all", action="store_true", help="đo hết mọi model (thí nghiệm), không dừng ở bản đạt đầu tiên")
    a = ap.parse_args()
    d = np.load(a.eval)
    clips, p_ref, thr = d["clip"].astype(np.float32), d["p"], float(d["threshold"])
    # bỏ clip trùng (vd. nhiều cửa sổ không thấy người cho cùng một clip -> một lỗi bị đếm hàng chục lần)
    _, keep = np.unique(clips.reshape(len(clips), -1), axis=0, return_index=True)
    keep = np.sort(keep)
    if len(keep) < len(clips):
        print(f"Bỏ {len(clips) - len(keep)} clip trùng -> đo {len(keep)} clip khác nhau")
        clips, p_ref = clips[keep], p_ref[keep]
    results, chosen = [], None
    for m in a.models:
        t0 = time.time()
        p = probs(m, clips)
        ref = p_ref[:len(p)]
        flips = int(((p >= thr) != (ref >= thr)).sum())
        err = np.abs(p - ref)
        near = np.abs(ref - thr) < 0.05
        print(f"{m}: {len(p)} clip | lệch P trung bình {err.mean():.4f}, lớn nhất {err.max():.4f} | "
              f"đổi nhãn @{thr:.2f}: {flips} ({100 * flips / len(p):.2f}%) | "
              f"trong đó clip sát ngưỡng (|P-{thr:.2f}|<0.05): {int(((p >= thr) != (ref >= thr))[near].sum())}/{int(near.sum())} "
              f"| {time.time() - t0:.0f} s")
        results.append((m, flips, float(err.mean())))
        if flips == 0 and err.mean() < 0.02 and chosen is None:
            chosen = m
            if not a.all:
                break
    if chosen is None:
        chosen = min(results, key=lambda r: (r[1], r[2]))[0]
    print("CHỌN:", chosen)
    with open(a.out, "w") as f:
        f.write(chosen)


if __name__ == "__main__":
    main()
