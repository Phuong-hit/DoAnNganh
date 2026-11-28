"""Thí nghiệm: việc GỘP GCN thành Conv1x1 dày (src/shiftgcn_npu.py) có làm lượng tử hóa INT8 tệ hơn không?

Xuất bản KHÔNG gộp (dịch khớp bằng Conv1x1 hoán vị 0/1, vì cv181x không hỗ trợ Gather) ra output/maix_convert/shiftgcn_unfused/, dùng CHUNG calib / test / eval
với bản gộp (output/maix_convert/shiftgcn/, tạo bởi  python export_maixcam_model.py --npu) để so cùng dữ liệu.
Sau đó trong WSL Ubuntu-22.04:
    bash /mnt/d/documentss/DoAnNganh/FallApp/fall_web/maixcam_app/exp_npu_fusion.sh
"""
import shutil

import numpy as np
import torch

from web.config import load_config, resolve
from web.engine import build_classifier
from src.shiftgcn_npu import to_npu_model


def main():
    src, dst = resolve("output/maix_convert/shiftgcn"), resolve("output/maix_convert/shiftgcn_unfused")
    cfg = load_config()
    clf = build_classifier({**cfg, "classifier": {**cfg["classifier"], "device": "cpu"}})
    fused, unfused = to_npu_model(clf.model, "fused"), to_npu_model(clf.model, "perm")
    ev = np.load(src / "eval.npz")
    x = torch.from_numpy(ev["clip"].astype(np.float32))
    with torch.no_grad():
        ref = clf.model.eval()(x)
        d_f, d_u = float((fused(x) - ref).abs().max()), float((unfused(x) - ref).abs().max())
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("calib", "test.npz", "eval.npz"):
        if (dst / name).exists():
            shutil.rmtree(dst / name) if (dst / name).is_dir() else (dst / name).unlink()
        (shutil.copytree if (src / name).is_dir() else shutil.copy)(src / name, dst / name)
    onnx = dst / "shiftgcn_unfused.onnx"
    torch.onnx.export(unfused, x[:1], str(onnx), input_names=["clip"], output_names=["logits"],
                      opset_version=13, dynamo=False)
    n = lambda m: sum(p.numel() for p in m.state_dict().values())   # noqa: E731
    print(f"Sai số logits so với model gốc: gộp {d_f:.2e} | không gộp {d_u:.2e}")
    print(f"Số giá trị lưu: gộp {n(fused):,} | không gộp {n(unfused):,}")
    print(f"ONNX: {onnx}")


if __name__ == "__main__":
    main()
