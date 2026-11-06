"""Kiểm thử bộ test chung (src/testset.py), Test Tool (test_tool.py), B0 và nạp Shift-GCN.

    python -m unittest tests.test_shared_testset

Dùng dataset giả nhỏ (không cần file thật trong data/). Phần cần PyTorch tự bỏ qua nếu chưa cài torch.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.testset import export_testset, load_testset, parse_video_id  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None
W = 17


def fake_dataset(path: Path, seed=0):
    """Dataset định dạng build_dataset_v1.py: 3 cửa sổ mỗi video, ngã = người đổ ngang nhanh."""
    rng = np.random.default_rng(seed)
    out = {}
    for split, subj in (("train", ["SBJ01", "SBJ02"]), ("val", ["SBJ03"]), ("test", ["SBJ04"])):
        X, y, vid = [], [], []
        for s in subj:
            for kind in ("Fall", "ADL"):
                for act in (1, 2):
                    v = f"{s}_{kind}_LOC1-ACT{act}-F-1"
                    for _ in range(3):
                        clip = rng.normal(0, 0.05, (W, 17, 3)).astype(np.float32)
                        clip[..., 1] += np.linspace(-1, 1, 17)[None, :]          # người đứng
                        if kind == "Fall":
                            clip[8:, :, 0], clip[8:, :, 1] = clip[8:, :, 1] * 1.0, 0.8   # nằm ngang, hạ thấp
                        clip[..., 2] = 1.0
                        X.append(clip)
                        y.append(int(kind == "Fall"))
                        vid.append(v)
        out[f"X_{split}"], out[f"y_{split}"] = np.stack(X), np.array(y, np.int64)
        out[f"video_{split}"] = np.array(vid)
    np.savez_compressed(path, **out)
    return path


class TestsetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = fake_dataset(self.tmp / "ds.npz")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parse_video_id(self):
        info = parse_video_id("SBJ13_Fall_LOC2-ACT10-F-1")
        self.assertEqual((info["subject"], info["kind"], info["loc"], info["act"], info["view"], info["trial"]),
                         ("SBJ13", "Fall", "LOC2", "ACT10", "F", "1"))
        self.assertEqual(parse_video_id("Subject 1_ADL_03")["subject"], "Subject 1")

    def test_export_load_and_tamper(self):
        out = export_testset(self.data, self.tmp / "ts", "test")
        ts = load_testset(out)
        man = ts["manifest"]
        self.assertEqual(man["counts"]["windows"], 12)
        self.assertEqual(man["counts"]["videos"], 4)
        self.assertEqual(man["counts"]["fall_videos"], 2)
        self.assertEqual(man["subjects"], ["SBJ04"])
        self.assertEqual(man["label_kind_mismatch"], [])
        with self.assertRaises(FileExistsError):                  # bộ test chung không bị ghi đè
            export_testset(self.data, self.tmp / "ts", "test")
        # sửa dữ liệu -> load phải báo
        with np.load(out / "testset.npz") as z:
            d = {k: z[k] for k in z.files}
        d["y"] = 1 - d["y"]
        np.savez_compressed(out / "testset.npz", **d)
        with self.assertRaises(ValueError):
            load_testset(out)

    def test_b0_only_run_and_save(self):
        import test_tool
        ts = export_testset(self.data, self.tmp / "ts", "test")
        b0 = self.tmp / "b0.json"
        b0.write_text(json.dumps({"params": {"angle_th": 50.0}}), encoding="utf-8")
        res = test_tool.run(testset=ts, models={"B0": str(b0)}, latency_runs=3)
        self.assertEqual(len(res["windows"]), 12)
        self.assertEqual(len(res["videos"]), 4)
        m = res["report"]["models"][0]
        self.assertEqual(m["role"], "B0")
        for k in ("prec", "rec", "f1", "tp", "fp", "fn", "tn"):
            self.assertIn(k, m["window"])
        self.assertIsNotNone(m["latency"]["mean_ms"])
        out = test_tool.save(res, self.tmp / "run")
        for name in ("predictions.csv", "video_predictions.csv", "comparison.csv", "errors.csv",
                     "by_activity.csv", "summary.json", "report.md"):
            self.assertTrue((out / name).is_file(), name)
        # lỗi trong errors phải khớp số FN + FP theo video
        v = m["video"]
        self.assertEqual(len(res["errors"]), v["fn"] + v["fp"])


@unittest.skipUnless(HAS_TORCH, "chưa cài torch")
class NeuralTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shiftgcn_team_checkpoint_loads(self):
        """Checkpoint định dạng train Shift-GCN của nhóm (model_state_dict, num_frames...) nạp được vào
        FallClassifier và cho cùng kết quả với Model gốc."""
        import torch
        from src.classifier import FallClassifier, build_model
        from src.model_shiftgcn import Model
        torch.manual_seed(0)
        ref = Model(num_class=2, num_point=17, in_channels=3, num_frames=W).eval()
        path = self.tmp / "shiftgcn_best.pt"
        torch.save({"epoch": 3, "model_state_dict": ref.state_dict(), "optimizer_state_dict": {},
                    "best_val_f1": 0.9, "num_frames": W, "num_point": 17, "in_channels": 3, "num_class": 2}, path)
        clf = FallClassifier(path, device="cpu", verbose=False)
        self.assertEqual((clf.arch, clf.window_size, clf.fall_threshold), ("shiftgcn", W, 0.5))
        X = np.random.default_rng(1).normal(size=(4, W, 17, 3)).astype(np.float32)
        X[..., 2] = 1.0
        with torch.no_grad():
            want = torch.softmax(ref(torch.from_numpy(X).permute(0, 3, 1, 2)), 1)[:, 1].numpy()
        np.testing.assert_allclose(clf.predict_batch(X), want, atol=1e-5)
        self.assertIsInstance(build_model("shiftgcn", num_frames=W), torch.nn.Module)

    def test_window_mismatch_is_reported(self):
        import test_tool
        from src.classifier import build_model, save_checkpoint
        ts = export_testset(fake_dataset(self.tmp / "ds.npz"), self.tmp / "ts", "test")
        ck = save_checkpoint(self.tmp / "lstm20.pt", build_model("lstm"), arch="lstm", window_size=20, fps=10.0)
        with self.assertRaises(ValueError):
            test_tool.run(testset=ts, models={"B1": str(ck)}, latency_runs=0)


if __name__ == "__main__":
    unittest.main()
