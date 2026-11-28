"""Đo nhanh tốc độ CPU của MaixCAM cho Shift-GCN (chạy 1 lần lúc khởi động nếu main.CPU_BENCH_ON_START = True).

Trả dict (in ra log + đưa vào /bench) để biết Shift-GCN chạy CPU chậm vì đâu:
  matmul_mflops      tốc độ nhân ma trận numpy (cỡ lớp 64 kênh: (289x64) @ (64x64))
  cv2_gemm_mflops    cùng phép nhân bằng OpenCV cv2.gemm (nếu có)
  shiftgcn_ref_ms    Shift-GCN numpy bản cũ ([C,T,V] + tensordot)
  shiftgcn_fast_ms   Shift-GCN numpy bản nhanh ([T,V,C] + matmul 2D liền mạch)
"""
import time

import numpy as np


def _time(fn, n, warm=1):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) * 1000.0 / n


def run(net, clips, budget_s=12.0):
    t_start = time.perf_counter()
    out = {"numpy_version": np.__version__}
    a = np.random.rand(289, 64).astype(np.float32)
    b = np.random.rand(64, 64).astype(np.float32)
    macs = 289 * 64 * 64
    ms = _time(lambda: a @ b, 20)
    out["matmul_ms"] = round(ms, 3)
    out["matmul_mflops"] = round(2 * macs / (ms / 1000.0) / 1e6, 1)
    try:
        import cv2
        ms = _time(lambda: cv2.gemm(a, b, 1.0, None, 0.0), 20)
        out["cv2_gemm_ms"] = round(ms, 3)
        out["cv2_gemm_mflops"] = round(2 * macs / (ms / 1000.0) / 1e6, 1)
    except Exception as e:                       # không có cv2 / gemm
        out["cv2_gemm_error"] = "%s: %s" % (type(e).__name__, e)
    clip = np.asarray(clips[0], dtype=np.float32)
    if hasattr(net, "logits_ref"):
        out["shiftgcn_fast_ms"] = round(_time(lambda: net.logits(clip), 5), 2)
        if time.perf_counter() - t_start < budget_s:
            out["shiftgcn_ref_ms"] = round(_time(lambda: net.logits_ref(clip), 3), 2)
        out["fast_vs_ref_max_diff"] = float(np.abs(net.logits(clip) - net.logits_ref(clip)).max())
    out["bench_s"] = round(time.perf_counter() - t_start, 2)
    return out
