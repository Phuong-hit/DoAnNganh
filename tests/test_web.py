"""Kiểm thử lớp web với backend pose giả (không cần ultralytics/model).

    python -m unittest discover tests
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web import csv_tools  # noqa: E402
from web.config import load_config  # noqa: E402
from web.engine import config_tag, extract_to_csv  # noqa: E402


def make_video(path: Path, secs=4.0, fps=30, size=(320, 240)):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(int(secs * fps)):
        w.write(np.full((size[1], size[0], 3), i % 255, np.uint8))
    w.release()


class WebTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        make_video(cls.tmp / "videos" / "fall" / "f 1.mp4", secs=4.0, fps=30)
        make_video(cls.tmp / "videos" / "adl" / "a1.mp4", secs=3.0, fps=25)
        cls.cfg = load_config(None, [
            "pose.backend=dummy",
            f"paths.videos={cls.tmp / 'videos'}",
            f"paths.metadata={cls.tmp / 'metadata.csv'}",
            f"paths.output={cls.tmp / 'output'}",
        ])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_extract_and_check(self):
        out = self.tmp / "x" / "f_1.csv"
        s = extract_to_csv(str(self.tmp / "videos/fall/f 1.mp4"), "f_1", out, self.cfg)
        self.assertEqual(s["frames"], 40)                      # 4 s @ 10 fps
        self.assertTrue(out.exists())
        self.assertEqual([p.name for p in out.parent.iterdir()], ["f_1.csv"])   # không còn thư mục tạm
        t, valid, joints = csv_tools.to_arrays(csv_tools.read_csv(out))
        np.testing.assert_allclose(t, np.arange(40) / 10, atol=1e-6)
        self.assertEqual(joints.shape, (40, 17, 3))
        self.assertTrue(np.isnan(joints[~valid]).all())         # quy ước CSVExporter: valid=0 -> ô trống
        rep = csv_tools.check_csv(out, self.cfg["check"])
        self.assertTrue(rep.ok, rep.to_text())
        long = csv_tools.to_long(out, self.tmp / "x" / "f_1.long.csv")
        self.assertEqual(len(csv_tools.read_csv(long)), 40 * 17)

    def test_check_rejects_bad_csv(self):
        out = self.tmp / "y" / "a1.csv"
        extract_to_csv(str(self.tmp / "videos/adl/a1.mp4"), "a1", out, self.cfg)
        df = csv_tools.read_csv(out)
        df.loc[3, "t"] = df.loc[2, "t"]                         # t trùng
        bad = self.tmp / "y" / "bad.csv"
        df.to_csv(bad, index=False)
        self.assertFalse(csv_tools.check_csv(bad).ok)

    def test_server(self):
        from web.server import create_app
        app = create_app(self.cfg)
        c = app.test_client()
        self.assertEqual(c.get("/").status_code, 200)
        r = c.post("/api/scan").get_json()
        self.assertEqual(r["videos"], 2)
        vids = {v["video_id"] for v in c.get("/api/videos").get_json()}
        self.assertEqual(vids, {"f_1", "a1"})
        job = c.post("/api/extract", json={"source": "all"}).get_json()
        self.assertEqual(job["total"], 2)
        import time
        for _ in range(100):
            jobs = c.get("/api/jobs").get_json()
            if jobs[-1]["status"] != "running":
                break
            time.sleep(0.1)
        self.assertEqual(jobs[-1]["done"], 2)
        out_csv = self.tmp / "output" / config_tag(self.cfg) / "csv" / "f_1.csv"
        self.assertTrue(out_csv.exists())
        sk = c.get("/api/skeleton/f_1").get_json()
        self.assertEqual(len(sk["t"]), 40)
        self.assertTrue(c.get("/api/check/f_1").get_json()["ok"])
        self.assertEqual(c.get("/download/f_1.csv").status_code, 200)
        self.assertEqual(c.get("/download/f_1.csv?format=long").status_code, 200)
        self.assertEqual(c.get("/download/nope.csv").status_code, 404)
        self.assertEqual(c.get("/video/f_1", headers={"Range": "bytes=0-9"}).status_code, 206)
        st = c.get("/api/status").get_json()
        self.assertEqual(st["skeletons"], 2)
        # luồng MJPEG từ file, không chờ thời gian thật
        resp = c.get("/stream?source=a1&realtime=0")
        body = b"".join(resp.response)
        self.assertEqual(body.count(b"--frame"), 30)            # 3 s @ 10 fps
        self.assertTrue(c.get("/api/live").get_json()["finished"])
        self.assertIn("error", c.post("/api/extract", json={"source": "0"}).get_json())


if __name__ == "__main__":
    unittest.main()
