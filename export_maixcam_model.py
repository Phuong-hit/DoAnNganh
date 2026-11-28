#!/usr/bin/env python3
"""Xuất checkpoint Shift-GCN (.pt, PyTorch) -> .npz cho MaixCAM (maixcam_app/src/shiftgcn_np.py, chạy bằng numpy).

MaixCAM không có PyTorch: board đọc trọng số từ .npz (chỉ mảng số, không pickle) và tự tính forward bằng numpy.
Ngưỡng + tham số hậu xử lý lấy từ configs/web.yaml (classifier.choices) để board chạy GIỐNG web/nhánh final.

  python export_maixcam_model.py                                        # model mặc định của web
  python export_maixcam_model.py --checkpoint model_classify/shiftgcn_best_4_20_11.pt
  python export_maixcam_model.py --check 300                            # kiểm tra numpy == PyTorch trên 300 clip thật

Ra: maixcam_app/model_classify/<tên>.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.config import load_config, resolve  # noqa: E402
from web.engine import build_classifier  # noqa: E402


def temporal_kernel(shift, num_frames: int) -> int:
    """Suy kernel thời gian từ bảng dịch của TemporalShift: độ dịch lớn nhất k//2 -> k = 2*max + 1."""
    idx = shift.idx.numpy().reshape(-1, num_frames)
    valid = shift.valid.numpy().reshape(-1, num_frames) > 0
    c, t = np.indices(idx.shape)
    off = np.abs(idx - (c * num_frames + t))[valid]
    return int(2 * off.max() + 1) if off.size else 1


def export(checkpoint: str | None = None, out: str | None = None, cfg=None) -> Path:
    cfg = cfg or load_config()
    if checkpoint:
        cfg["classifier"]["checkpoint"] = checkpoint
    clf = build_classifier({**cfg, "classifier": {**cfg["classifier"], "device": "cpu"}})
    if clf is None or clf.arch != "shiftgcn":
        raise SystemExit(f"Chỉ hỗ trợ Shift-GCN (đang chọn: {clf and clf.name}).")
    m = clf.model
    arrays = {f"w/{k}": v.detach().cpu().numpy().astype(np.float32)
              for k, v in m.state_dict().items() if not k.endswith("num_batches_tracked")}
    blocks = list(m.blocks)
    meta = dict(
        arch="shiftgcn", source=str(clf.checkpoint_path.name),
        num_class=m.num_class, num_point=m.num_point, in_channels=m.in_channels, num_frames=m.num_frames,
        channels=[b.gcn.out_channels for b in blocks],
        strides=[b.tcn.stride for b in blocks],
        kernel=temporal_kernel(blocks[0].tcn.shift_in, m.num_frames),
        residual=[b.residual is not None for b in blocks],
        bn_eps=float(m.data_bn.eps),
        threshold=float(clf.fall_threshold),
        window_size=int(clf.window_size),
        stride_frames=int(clf.stride_frames),
        postprocess=dict(clf.post_cfg) if clf.post is not None else None,
        fps=10.0,
    )
    out_path = Path(out) if out else resolve("maixcam_app/model_classify") / (clf.checkpoint_path.stem + ".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, meta=np.array(json.dumps(meta)), **arrays)
    return out_path


def check(npz: Path, n: int) -> None:
    """So xác suất ngã: numpy (board) vs PyTorch (PC) trên các clip thật trong output/npz."""
    import glob
    import importlib.util
    from src.features import normalize_clip
    cfg = load_config()
    clf = build_classifier({**cfg, "classifier": {**cfg["classifier"], "device": "cpu"}})
    spec = importlib.util.spec_from_file_location("shiftgcn_np", resolve("maixcam_app/src/shiftgcn_np.py"))
    snp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snp)
    net = snp.ShiftGCN.load(str(npz))
    W = clf.window_size
    clips = []
    for f in sorted(glob.glob(str(resolve("output/npz/npz/*.npz")))):
        d = np.load(f, allow_pickle=True)
        seq, mask = d["seq"], d["mask"].astype(bool)
        aspect = float(d["aspect"]) if "aspect" in d.files else 1.0
        for e in range(W - 1, len(seq), 3):
            clips.append(normalize_clip(seq[e - W + 1:e + 1], mask[e - W + 1:e + 1], aspect))
            if len(clips) >= n:
                break
        if len(clips) >= n:
            break
    if not clips:
        clips = [np.random.default_rng(0).normal(size=(W, 17, 3)).astype(np.float32) for _ in range(n)]
    p_pt = clf.predict_batch(np.stack(clips))
    p_np = np.array([net.predict(c) for c in clips])
    diff = np.abs(p_pt - p_np)
    flips = int(((p_pt >= clf.fall_threshold) != (p_np >= clf.fall_threshold)).sum())
    print(f"Kiểm tra {len(clips)} clip: |P_numpy - P_torch| max {diff.max():.2e}, mean {diff.mean():.2e} | "
          f"khác nhãn (ngưỡng {clf.fall_threshold}): {flips}")


def export_npu(out_dir: Path, n_calib: int = 200, n_eval: int = 600) -> Path:
    """ONNX của Shift-GCN viết lại cho NPU (src/shiftgcn_npu.py) + dữ liệu cho TPU-MLIR:
    calib/*.npz (hiệu chỉnh INT8), test.npz, eval.npz (clip + P(ngã) PyTorch để đo sai số sau lượng tử hóa)."""
    import glob
    import torch
    from src.features import normalize_clip
    from src.shiftgcn_npu import max_abs_diff, to_npu_model
    cfg = load_config()
    clf = build_classifier({**cfg, "classifier": {**cfg["classifier"], "device": "cpu"}})
    npu = to_npu_model(clf.model)
    W = clf.window_size
    clips, falls = [], []
    for f in sorted(glob.glob(str(resolve("output/npz/npz/*.npz")))):
        d = np.load(f, allow_pickle=True)
        seq, mask = d["seq"], d["mask"].astype(bool)
        aspect = float(d["aspect"]) if "aspect" in d.files else 1.0
        for e in range(W - 1, len(seq), 2):
            clips.append(normalize_clip(seq[e - W + 1:e + 1], mask[e - W + 1:e + 1], aspect))
    clips = np.stack(clips).astype(np.float32)
    x_all = np.ascontiguousarray(clips.transpose(0, 3, 1, 2))              # [N, 3, T, V]
    with torch.no_grad():
        p_all = torch.softmax(npu(torch.from_numpy(x_all)), 1)[:, 1].numpy()
    diff = max_abs_diff(clf.model, npu, clips)
    rng = np.random.default_rng(0)
    fall_idx = np.where(p_all >= clf.fall_threshold)[0]
    other = np.where(p_all < clf.fall_threshold)[0]
    pick = lambda idx, k: rng.choice(idx, size=min(k, len(idx)), replace=False)    # noqa: E731
    calib = np.concatenate([pick(fall_idx, n_calib // 2), pick(other, n_calib - min(n_calib // 2, len(fall_idx)))])
    ev = np.concatenate([pick(fall_idx, n_eval // 2), pick(other, n_eval // 2)])

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "shiftgcn_npu.onnx"
    torch.onnx.export(npu, torch.from_numpy(x_all[:1]), str(onnx_path), input_names=["clip"],
                      output_names=["logits"], opset_version=13, dynamo=False)
    cdir = out_dir / "calib"
    cdir.mkdir(exist_ok=True)
    for old in cdir.glob("*.npz"):
        old.unlink()
    for k, i in enumerate(calib):
        np.savez(cdir / f"{k:03d}.npz", clip=x_all[i:i + 1])
    np.savez(out_dir / "test.npz", clip=x_all[fall_idx[0]:fall_idx[0] + 1] if len(fall_idx) else x_all[:1])
    np.savez(out_dir / "eval.npz", clip=x_all[ev], p=p_all[ev], threshold=np.float32(clf.fall_threshold))
    # 8 clip (4 ngã + 4 không) để MaixCAM tự kiểm tra NPU lúc khởi động (clip dạng [T, V, 3] như normalize_clip)
    # chọn clip xa ngưỡng (ngã P >= 0.9, không ngã P <= 0.5) để sai số nhỏ của NPU không làm đổi nhãn khi tự kiểm tra
    chk = np.concatenate([pick(np.where(p_all >= 0.9)[0], 4), pick(np.where(p_all <= 0.5)[0], 4)])
    np.savez(resolve("maixcam_app/model_classify/shiftgcn_npu_check.npz"),
             clip=clips[chk].astype(np.float32), p=p_all[chk].astype(np.float64))
    print(f"ONNX {onnx_path} | sai số so với model gốc {diff:.2e} | calib {len(calib)} clip | eval {len(ev)} clip "
          f"({int((p_all[ev] >= clf.fall_threshold).sum())} ngã)")
    return onnx_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", type=int, default=300, help="số clip kiểm tra numpy == PyTorch (0 = bỏ qua)")
    ap.add_argument("--npu", action="store_true",
                    help="thêm: xuất ONNX cho NPU + dữ liệu TPU-MLIR vào output/maix_convert/shiftgcn/ "
                         "(rồi chạy maixcam_app/convert_shiftgcn.sh trong WSL Ubuntu-22.04)")
    args = ap.parse_args(argv)
    if args.npu:
        export_npu(resolve("output/maix_convert/shiftgcn"))
    path = export(args.checkpoint, args.out)
    meta = json.loads(str(np.load(path)["meta"]))
    print(f"Đã xuất {path} ({path.stat().st_size / 1024:.0f} KB) | ngưỡng {meta['threshold']} | "
          f"cửa sổ {meta['window_size']} | hậu xử lý {'bật' if meta['postprocess'] else 'tắt'}")
    if args.check:
        check(path, args.check)


if __name__ == "__main__":
    main()
