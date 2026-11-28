"""Bộ phát hiện ngã chạy trên MaixCAM (maixcam_app/src/fall_detect.py, numpy) phải cho KẾT QUẢ GIỐNG web
(src/classifier.py FallClassifier + hậu xử lý nhánh final) trên cùng chuỗi pose.

  python -m pytest tests/test_maixcam_fall.py -q      hoặc      python tests/test_maixcam_fall.py
Cần: maixcam_app/model_classify/<model>.npz (python export_maixcam_model.py) và output/npz/npz/*.npz.
"""
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_estimate import PoseResult  # noqa: E402


def board_pkg():
    """Nạp maixcam_app/src dưới tên 'maixsrc' (tránh trùng gói src/ của PC)."""
    if "maixsrc" in sys.modules:
        return sys.modules["maixsrc"]
    d = ROOT / "maixcam_app" / "src"
    spec = importlib.util.spec_from_file_location("maixsrc", d / "__init__.py", submodule_search_locations=[str(d)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["maixsrc"] = mod
    spec.loader.exec_module(mod)
    return mod


def npz_model():
    files = sorted((ROOT / "maixcam_app" / "model_classify").glob("*.npz"))
    return files[0] if files else None


def poses_from_npz(f):
    d = np.load(f, allow_pickle=True)
    seq, mask = d["seq"], d["mask"].astype(bool)
    valid = d["valid"].astype(bool) if "valid" in d.files else mask.any(1)
    aspect = float(d["aspect"]) if "aspect" in d.files else 1.0
    out = []
    for i in range(len(seq)):
        j = np.full((17, 3), np.nan, np.float32)
        j[:, 2] = 0.0
        j[mask[i], :2] = seq[i, mask[i], :2]
        j[mask[i], 2] = 1.0
        out.append(PoseResult(i, i / 10, bool(valid[i]), j, int(round(aspect * 1000)), 1000))
    return out


def test_normalize_matches_pc():
    from src.features import normalize_clip as pc_norm
    fd = importlib.import_module("maixsrc.fall_detect") if board_pkg() else None
    rng = np.random.default_rng(0)
    for _ in range(50):
        seq = rng.random((17, 17, 3)).astype(np.float32)
        mask = rng.random((17, 17)) > 0.3
        seq[~mask] = 0
        a = float(rng.uniform(0.5, 2.0))
        assert np.allclose(pc_norm(seq, mask, a), fd.normalize_clip(seq, mask, a), atol=1e-5)


def test_speed_gate_matches_pc():
    """Cổng tốc độ trên board (fall_detect.speed_ok) == src/motion.py SpeedGate (không keep_if_p_above)."""
    from src.motion import SpeedGate, motion_features
    fd = importlib.import_module("maixsrc.fall_detect") if board_pkg() else None
    f = ROOT / "data" / "testsets" / "v1_lying" / "testset.npz"
    if not f.exists():
        import pytest
        pytest.skip("thiếu data/testsets/v1_lying")
    X = np.load(f, allow_pickle=True)["X"]
    gate = dict(shoulder_drop=2.0, rotation=200.0)
    pc = SpeedGate(True, 2.0, 200.0, None).passes(motion_features(X, 10.0), np.zeros(len(X)))
    board = np.array([fd.speed_ok(x, gate, 10.0) for x in X])
    assert 0 < board.sum() < len(X) and (pc == board).all()


def test_board_detector_matches_web():
    model = npz_model()
    files = sorted(glob.glob(str(ROOT / "output" / "npz" / "npz" / "*.npz")))
    if model is None or not files:
        import pytest
        pytest.skip("thiếu model .npz hoặc dữ liệu output/npz")
    board_pkg()
    fd = importlib.import_module("maixsrc.fall_detect")
    from web.config import load_config
    from web.engine import build_classifier
    cfg = load_config()
    web_clf = build_classifier({**cfg, "classifier": {**cfg["classifier"], "device": "cpu"}})
    det = fd.FallDetector.load(str(model))
    assert det.threshold == web_clf.fall_threshold and det.window_size == web_clf.window_size

    files = [f for f in files if "Fall" in f][:10] + [f for f in files if "ADL" in f][:6]
    total = 0
    for f in files:
        web_clf.reset()
        det.reset()
        for p in poses_from_npz(f):
            a, b = web_clf.update(p), det.update(p)
            assert (a is None) == (b is None), f
            if a is None:
                continue
            total += 1
            assert a["label"] == b["label"], (f, p.frame_idx)
            assert abs(a["probability"] - b["prob"]) < 1e-4, (f, p.frame_idx)
    assert total > 100
    return total


if __name__ == "__main__":
    test_normalize_matches_pc()
    test_speed_gate_matches_pc()
    n = test_board_detector_matches_web()
    print(f"OK: board == web trên {n} khung")
