"""Kiểm thử phần phát hiện ngã: NPZ, dataset cửa sổ, đồ thị/chuẩn hóa, bộ phân loại.

    python -m unittest discover tests

Các test cần PyTorch tự bỏ qua nếu chưa cài torch.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import NUM_KEYPOINTS  # noqa: E402
from src.features import flip_clip, normalize_clip  # noqa: E402
from src.graph import build_adjacency  # noqa: E402
from web.config import load_config  # noqa: E402
from web.dataset import build_dataset, load_dataset  # noqa: E402
from web.engine import extract_to_csv, load_npz, npz_dir  # noqa: E402
from web.videos import scan_videos  # noqa: E402

from test_web import make_video  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None


class FeatureTest(unittest.TestCase):
    def test_graph(self):
        A = build_adjacency()
        self.assertEqual(A.shape, (3, NUM_KEYPOINTS, NUM_KEYPOINTS))
        self.assertTrue(np.isfinite(A).all() and (A >= 0).all())

    def test_normalize_and_flip(self):
        rng = np.random.default_rng(0)
        seq = rng.uniform(0.2, 0.8, (20, 17, 3)).astype(np.float32)
        mask = np.ones((20, 17), bool)
        mask[5, 3] = False
        clip = normalize_clip(seq, mask, aspect=4 / 3)
        self.assertEqual(clip.shape, (20, 17, 3))
        self.assertEqual(clip[5, 3].tolist(), [0.0, 0.0, 0.0])          # ô thiếu = 0, mask = 0
        np.testing.assert_array_equal(flip_clip(flip_clip(clip)), clip)
        self.assertFalse(normalize_clip(seq, np.zeros((20, 17), bool)).any())


class DatasetTest(unittest.TestCase):
    """Dummy backend: chu kỳ 80 khung, ngã ở khung 40..48 (t = 4.0..4.8 s @ 10 fps)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        make_video(cls.tmp / "videos" / "fall" / "f1.mp4", secs=7.0, fps=30)
        make_video(cls.tmp / "videos" / "adl" / "a1.mp4", secs=5.0, fps=25)
        cls.cfg = load_config(None, [
            "pose.backend=dummy",
            f"paths.videos={cls.tmp / 'videos'}",
            f"paths.metadata={cls.tmp / 'metadata.csv'}",
            f"paths.output={cls.tmp / 'output'}",
            "classifier.checkpoint=null",
        ])
        scan_videos(cls.tmp / "videos", cls.tmp / "metadata.csv")
        meta = pd.read_csv(cls.tmp / "metadata.csv", dtype=str)
        meta.loc[meta.video_id == "f1", ["fall_start_s", "fall_end_s", "split", "subject_id"]] = \
            ["4.0", "4.8", "train", "s1"]
        meta.loc[meta.video_id == "a1", ["split", "subject_id"]] = ["val", "s2"]
        meta.to_csv(cls.tmp / "metadata.csv", index=False)
        from web.engine import csv_dir
        for vid in ("f1", "a1"):
            label = "fall" if vid == "f1" else "adl"
            extract_to_csv(str(cls.tmp / "videos" / label / f"{vid}.mp4"), vid,
                           csv_dir(cls.cfg) / f"{vid}.csv", cls.cfg)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_npz(self):
        d = load_npz(npz_dir(self.cfg) / "f1.npz")
        self.assertEqual(d["seq"].shape, (70, 17, 3))
        self.assertEqual(d["mask"].shape, (70, 17))
        self.assertAlmostEqual(d["fps"], 10.0, places=3)
        self.assertAlmostEqual(d["aspect"], 320 / 240, places=5)
        self.assertEqual(d["video_id"], "f1")
        self.assertFalse(d["mask"][~d["valid"]].any())                   # khung không người: mask False
        self.assertEqual(float(np.abs(d["seq"][~d["mask"]]).sum()), 0.0)  # ô thiếu = 0 (quy ước skeleton_io)

    def test_build_dataset(self):
        path, stats, notes = build_dataset(self.cfg)
        ds = load_dataset(path)
        self.assertEqual(ds["window_size"], 20)
        self.assertEqual(ds["X"].shape[1:], (20, 17, 3))
        self.assertEqual(len(ds["X"]), len(ds["y"]))
        self.assertEqual(len(ds["y"]), len(ds["video"]))
        self.assertGreater(stats["fall"], 0)
        self.assertTrue((ds["y"][ds["video"] == "a1"] == 0).all())
        self.assertEqual(set(ds["split"][ds["video"] == "f1"]), {"train"})
        self.assertEqual(set(ds["subject"][ds["video"] == "a1"]), {"s2"})
        fall_rows = ds["y"] == 1
        self.assertTrue(set(ds["video"][fall_rows]) == {"f1"})
        self.assertFalse(notes, notes)

    def test_server_dataset_and_predict_without_model(self):
        from web.server import create_app
        c = create_app(self.cfg).test_client()
        self.assertEqual(c.get("/").status_code, 200)
        self.assertIn("chưa chọn model", c.get("/api/classifier").get_json()["error"])
        self.assertEqual(c.post("/api/classifier", json={"checkpoint": "models/khong_co.pt"}).status_code, 400)
        off = c.post("/api/classifier", json={"checkpoint": None})
        self.assertEqual(off.status_code, 200)
        self.assertIsNone(off.get_json()["name"])
        r = c.post("/api/build-dataset").get_json()
        self.assertGreater(r["windows"], 0)
        self.assertEqual(c.get("/api/predict/f1").status_code, 400)   # chưa có checkpoint
        v = {x["video_id"]: x for x in c.get("/api/videos").get_json()}
        self.assertTrue(v["f1"]["has_npz"])

    # ------------------------------------------------------------------ cần torch
    @unittest.skipUnless(HAS_TORCH, "chưa cài torch")
    def test_classifier_end_to_end(self):
        import torch
        from src.classifier import FallClassifier, build_model, save_checkpoint
        from src.stgcn_inference import STGCNInference
        from web.engine import predict_video

        torch.manual_seed(0)
        model = build_model("stgcn")
        self.assertEqual(tuple(model(torch.zeros(2, 3, 20, 17)).shape), (2, 2))
        ck = save_checkpoint(self.tmp / "m" / "stgcn.pt", model, arch="stgcn", window_size=20, fps=10.0)

        # checkpoint kiểu train.py cũ (không có "arch")
        legacy = self.tmp / "m" / "legacy.pt"
        torch.save({"state": model.state_dict(), "in_channels": 3, "num_classes": 2}, legacy)
        old = STGCNInference(legacy)
        self.assertEqual(old.arch, "stgcn")

        clf = FallClassifier(ck, stride_frames=5, verbose=False)
        item = load_npz(npz_dir(self.cfg) / "f1.npz")
        from src.pose_estimate import PoseResult
        outs = []
        for i in range(len(item["seq"])):
            j = item["seq"][i].copy()
            j[~item["mask"][i], :2] = np.nan
            outs.append(clf.update(PoseResult(i, float(item["t"][i]), bool(item["valid"][i]), j, 320, 240)))
        got = [o for o in outs if o is not None]
        self.assertEqual(len(got), len(range(19, 70, 5)))
        self.assertTrue(all(0.0 <= o["probability"] <= 1.0 for o in got))
        pv = predict_video(clf, item)
        self.assertEqual(len(pv["p"]), len(got))
        np.testing.assert_allclose(pv["p"], [o["probability"] for o in got], atol=1e-4)  # cùng cách chuẩn hóa

        cfg = load_config(None, [f"{k}={v}" for k, v in (
            ("pose.backend", "dummy"), ("paths.videos", self.tmp / "videos"),
            ("paths.metadata", self.tmp / "metadata.csv"), ("paths.output", self.tmp / "output"),
            ("classifier.checkpoint", ck), ("classifier.device", "cpu"))])
        from web.server import create_app
        c = create_app(cfg).test_client()
        self.assertEqual(len(c.get("/api/predict/f1").get_json()["p"]), len(got))
        body = b"".join(c.get("/stream?source=a1&realtime=0&classify=1").response)
        self.assertEqual(body.count(b"--frame"), 50)
        live = c.get("/api/live").get_json()
        self.assertIn("fall_prob", live)
        self.assertEqual(live["classifier"], "stgcn:stgcn")


if __name__ == "__main__":
    unittest.main()
