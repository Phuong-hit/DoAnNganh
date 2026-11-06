"""Kiểm thử phần huấn luyện / đánh giá (B1).

    python -m unittest discover tests

Phần numpy (đọc dataset, tăng cường, chỉ số) luôn chạy. Phần cần PyTorch (LSTM, train.py,
evaluate.py, nạp best_stgcn.pt) tự bỏ qua nếu chưa cài torch.
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

from src.train_utils import (best_threshold, flip_batch, jitter_batch, load_splits, metrics,  # noqa: E402
                             roc_auc, split_subjects, video_level)

HAS_TORCH = importlib.util.find_spec("torch") is not None
DATASET = ROOT / "data" / "dataset.npz"
STGCN_CKPT = ROOT / "models" / "best_stgcn.pt"


def fake_clips(n, t=20, seed=0):
    """Cửa sổ giả đã chuẩn hóa; lớp 1 có hông 'rơi' xuống theo thời gian để model học được."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.3, (n, t, 17, 3)).astype(np.float32)
    X[..., 2] = 1.0
    y = (np.arange(n) % 3 == 0).astype(np.int64)
    X[y == 1, :, :, 1] += np.linspace(0, 2, t)[None, :, None]
    return X, y


class TrainUtilsTest(unittest.TestCase):
    def test_flip_and_jitter_keep_missing_zero(self):
        X, _ = fake_clips(4)
        X[0, 3, 5] = 0.0                                    # khớp thiếu
        f = flip_batch(X)
        np.testing.assert_array_equal(flip_batch(f), X)
        self.assertEqual(f[0, 3, 6].tolist(), [0.0, 0.0, 0.0])   # khớp 5 (vai trái) đổi sang vị trí 6
        j = jitter_batch(X, std=0.05, rng=np.random.default_rng(1))
        self.assertEqual(j[0, 3, 5].tolist(), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(j[..., 2], X[..., 2])

    def test_metrics_threshold_auc(self):
        m = metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["f1"], 0.5)
        y = np.array([0, 0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.35, 0.4, 0.9])
        self.assertEqual(roc_auc(y, p), 1.0)
        t = best_threshold(y, p)
        self.assertTrue(0.35 < t <= 0.4)
        v = video_level(np.array([0, 1, 0, 0]), np.array([0.1, 0.9, 0.2, 0.7]), ["a", "a", "b", "b"], 0.5)
        self.assertEqual((v["tp"], v["fp"]), (1, 1))

    def test_split_subjects_matches_code1(self):
        tr, va, te = split_subjects(["Subject 1", "Subject 2", "Subject 3", "Subject 4"])
        self.assertEqual((len(tr), len(va), len(te)), (2, 1, 1))
        self.assertEqual((sorted(tr), va, te), (["Subject 2", "Subject 4"], {"Subject 1"}, {"Subject 3"}))

    @unittest.skipUnless(DATASET.exists(), "không có data/dataset.npz")
    def test_load_code1_dataset(self):
        d = load_splits(DATASET)
        self.assertEqual(d["format"], "code_1")
        self.assertEqual(d["window_size"], 20)
        self.assertEqual(len(d["train"][0]), 1089)
        self.assertEqual({v.split("_")[0] for v in d["test"][2]}, {"Subject 3"})

    def test_load_web_dataset_auto_split(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            X, y = fake_clips(40)
            vid = np.array([f"Subject {i % 4 + 1}_Fall_{i:02d}" for i in range(40)])
            np.savez(tmp / "w.npz", X=X, y=y, video=vid, subject=np.array([""] * 40),
                     split=np.array([""] * 40), fps=10.0, window_size=20)
            d = load_splits(tmp / "w.npz")
            self.assertEqual(d["format"], "web")
            self.assertEqual(d["auto_split"]["test"], ["Subject 3"])
            self.assertEqual(sum(len(d[s][1]) for s in ("train", "val", "test")), 40)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(HAS_TORCH, "chưa cài torch")
class TorchTrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        Xtr, ytr = fake_clips(96, seed=0)
        Xva, yva = fake_clips(30, seed=1)
        Xte, yte = fake_clips(30, seed=2)
        vids = lambda n, s: np.array([f"{s}_Fall_{i % 5}" for i in range(n)])  # noqa: E731
        np.savez(cls.tmp / "ds.npz", X_train=Xtr, y_train=ytr, video_train=vids(96, "Subject 2"),
                 X_val=Xva, y_val=yva, video_val=vids(30, "Subject 1"),
                 X_test=Xte, y_test=yte, video_test=vids(30, "Subject 3"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_lstm_forward_variants(self):
        import torch
        from src.classifier import build_model
        x = torch.randn(4, 3, 20, 17)
        x[:, 2] = (x[:, 2] > 0).float()
        for kw in ({}, {"bidirectional": True}, {"velocity": False, "layers": 1}):
            m = build_model("lstm", **kw).eval()
            self.assertEqual(tuple(m(x).shape), (4, 2), kw)
        m = build_model("lstm")
        m.train()
        m(x).sum().backward()                                    # gradient đi qua cả phần vận tốc

    def test_train_lstm_and_evaluate(self):
        import train as train_mod
        import evaluate as eval_mod
        from src.classifier import FallClassifier
        out = self.tmp / "lstm.pt"
        args = train_mod.build_parser().parse_args(
            ["--arch", "lstm", "--data", str(self.tmp / "ds.npz"), "--out", str(out), "--epochs", "3",
             "--patience", "5", "--device", "cpu", "--model-kw", "hidden=32", "--model-kw", "embed=32"])
        rep = train_mod.train(args)
        self.assertTrue(out.exists())
        self.assertEqual(rep["arch"], "lstm")
        report = json.loads((self.tmp / "lstm.report.json").read_text(encoding="utf-8"))
        self.assertIn("test_05", report)
        self.assertTrue((self.tmp / "lstm.history.csv").exists())
        clf = FallClassifier(out, device="cpu", verbose=False)
        self.assertEqual((clf.arch, clf.window_size), ("lstm", 20))
        self.assertAlmostEqual(clf.fall_threshold, rep["threshold"])
        self.assertEqual(clf.meta["model_kwargs"], {"hidden": 32, "embed": 32})
        _, rows = eval_mod.evaluate([out], self.tmp / "ds.npz", device="cpu", runs=3)
        self.assertEqual(rows[0]["arch"], "lstm")

    @unittest.skipUnless(STGCN_CKPT.exists() and DATASET.exists(), "không có best_stgcn.pt / dataset.npz")
    def test_code1_stgcn_checkpoint(self):
        """best_stgcn.pt (huấn luyện trên Colab, lưu từ GPU, không có 'arch') nạp được trên CPU."""
        from src.classifier import FallClassifier
        clf = FallClassifier(STGCN_CKPT, device="cpu", verbose=False)
        self.assertEqual(clf.arch, "stgcn")
        X, y, _ = load_splits(DATASET)["test"]
        p = clf.predict_batch(X)
        self.assertEqual(p.shape, (len(y),))
        self.assertGreater(roc_auc(y, p), 0.5)


if __name__ == "__main__":
    unittest.main()


class _StubClf:
    """Bộ phân loại giả (không cần torch) để kiểm thử trang so sánh."""
    def __init__(self, path, device=None):
        self.arch = "lstm" if "lstm" in Path(path).name else "stgcn"
        self.name, self.window_size, self.device = f"{self.arch}:{Path(path).stem}", 20, "cpu"
        self.fall_threshold = 0.45 if self.arch == "lstm" else 0.5
        self.num_params, self.meta = (275884 if self.arch == "lstm" else 3_000_000), {"epoch": 3}
        self.seed = 1 if self.arch == "lstm" else 2

    def predict_batch(self, X):
        hip = np.asarray(X)[:, :, 11:13, 1].mean(axis=2)
        drop = hip[:, -5:].mean(1) - hip[:, :5].mean(1)
        noise = np.random.default_rng(self.seed).normal(0, 0.3, len(hip))
        return np.clip(1 / (1 + np.exp(-(drop * 3 - 1))) + noise, 0, 1)


@unittest.skipUnless(DATASET.exists(), "không có data/dataset.npz")
class CompareTest(unittest.TestCase):
    def test_compare_engine_and_report(self):
        from web.compare import compare_models, render_report, to_csv_rows
        tmp = Path(tempfile.mkdtemp())
        try:
            a, b = tmp / "best_stgcn.pt", tmp / "best_lstm.pt"
            a.write_bytes(b"x"), b.write_bytes(b"y")
            r = compare_models([a, b], DATASET, "test", runs=2, n_boot=100, loader=_StubClf)
            self.assertEqual(r["dataset"]["windows"], 467)
            self.assertEqual([m["name"] for m in r["models"]], ["best_stgcn", "best_lstm"])
            self.assertEqual(len(r["pairs"]), 1)
            self.assertEqual(len(r["videos"]), 43)
            m = r["models"][1]
            self.assertEqual(m["roc"][0], [0.0, 0.0])
            self.assertEqual(m["roc"][-1], [1.0, 1.0])
            self.assertEqual(m["atthr"]["tp"] + m["atthr"]["fn"], 95)
            self.assertTrue(r["verdict"])
            v = r["videos"][0]
            self.assertEqual(len(r["windows"][v["video"]]["y"]), v["windows"])
            json.dumps(r)                                        # JSON được
            html = render_report(r, standalone=True)
            self.assertIn("best_lstm", html)
            self.assertNotIn("/api/compare\", {method", html.split("if (!STANDALONE)")[0][-200:])
            self.assertEqual(len(to_csv_rows(r)), 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_compare_routes(self):
        import web.compare as cmp
        import web.server as srv
        from web.config import load_config
        orig = cmp.compare_models
        cmp.compare_models = lambda *a, **k: orig(*a, **{**k, "loader": _StubClf, "n_boot": 50, "runs": 1})
        try:
            app = srv.create_app(load_config(None, ["pose.backend=dummy", "classifier.checkpoint=null"]))
            c = app.test_client()
            self.assertEqual(c.get("/compare").status_code, 200)
            self.assertEqual(c.get("/compare/export.csv").status_code, 404)        # chưa chạy
            r = c.post("/api/compare", json={"checkpoints": ["models/best_stgcn.pt"], "data": "data/dataset.npz"})
            self.assertEqual(r.status_code, 200, r.get_json())
            self.assertEqual(r.get_json()["models"][0]["name"], "best_stgcn")
            for fmt in ("csv", "json", "html"):
                self.assertEqual(c.get(f"/compare/export.{fmt}").status_code, 200)
            self.assertEqual(c.post("/api/compare", json={"checkpoints": ["../x.pt"], "data": "data/dataset.npz"}).status_code, 400)
            self.assertEqual(c.post("/api/compare", json={"checkpoints": ["models/best_stgcn.pt"], "data": "/etc/passwd"}).status_code, 400)
        finally:
            cmp.compare_models = orig


@unittest.skipUnless(DATASET.exists(), "không có data/dataset.npz")
class LabelCleanTest(unittest.TestCase):
    def test_fix_static_lying(self):
        import check_labels
        tmp = Path(tempfile.mkdtemp())
        try:
            out = tmp / "clean.npz"
            check_labels.main([str(DATASET), "--fix", "--out", str(out)])
            a, b = np.load(DATASET), np.load(out)
            self.assertLess(int(b["y_train"].sum()), int(a["y_train"].sum()))
            self.assertTrue((b["y_train"] <= a["y_train"]).all())           # chỉ đổi 1 -> 0
            np.testing.assert_array_equal(a["y_test"], b["y_test"])         # test giữ nguyên
            np.testing.assert_array_equal(a["X_train"], b["X_train"])
            f = check_labels.window_features(a["X_train"])
            changed = a["y_train"] != b["y_train"]
            self.assertTrue((f["horiz"][changed] >= 0.8).all())             # chỉ cửa sổ nằm ngang
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(HAS_TORCH, "chưa cài torch")
class MergeValTest(unittest.TestCase):
    def test_merge_val_from_report(self):
        import train as train_mod
        tmp = Path(tempfile.mkdtemp())
        try:
            Xtr, ytr = fake_clips(60, seed=0)
            Xva, yva = fake_clips(30, seed=1)
            Xte, yte = fake_clips(30, seed=2)
            v = lambda n, s: np.array([f"{s}_Fall_{i % 4}" for i in range(n)])  # noqa: E731
            np.savez(tmp / "ds.npz", X_train=Xtr, y_train=ytr, video_train=v(60, "Subject 2"),
                     X_val=Xva, y_val=yva, video_val=v(30, "Subject 1"),
                     X_test=Xte, y_test=yte, video_test=v(30, "Subject 3"))
            common = ["--arch", "lstm", "--data", str(tmp / "ds.npz"), "--device", "cpu",
                      "--model-kw", "hidden=16", "--model-kw", "embed=16"]
            rep1 = train_mod.train(train_mod.build_parser().parse_args(
                common + ["--out", str(tmp / "a.pt"), "--epochs", "4", "--patience", "10"]))
            rep2 = train_mod.train(train_mod.build_parser().parse_args(
                common + ["--out", str(tmp / "b.pt"), "--merge-val", "--from-report", str(tmp / "a.report.json")]))
            self.assertTrue(rep2["merged_val"])
            self.assertEqual(rep2["best_epoch"], rep1["best_epoch"])
            self.assertAlmostEqual(rep2["threshold"], rep1["threshold"])
            self.assertIsNone(rep2["val_05"])
            from src.classifier import FallClassifier
            clf = FallClassifier(tmp / "b.pt", device="cpu", verbose=False)
            self.assertTrue(clf.meta["merged_val"])
            with self.assertRaises(SystemExit):                              # thiếu số epoch
                train_mod.train(train_mod.build_parser().parse_args(common + ["--out", str(tmp / "c.pt"), "--merge-val"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
