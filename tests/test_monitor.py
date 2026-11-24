"""Kiểm thử giám sát nền + lưu event + nguồn maix:// (MaixCAM giả lập bằng HTTP server cục bộ).

    python -m unittest tests.test_monitor -v
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pose_estimate import PoseResult, empty_joints  # noqa: E402
from web.events import EventRecorder, EventStore  # noqa: E402
from web.maixcam import MaixPoseSource, maix_url, pose_from_json  # noqa: E402


def fake_pose(i, t, valid=True):
    if not valid:
        return PoseResult.empty(i, t, 320, 240)
    j = empty_joints()
    for k in range(17):
        j[k] = (0.3 + 0.01 * k, 0.2 + 0.03 * k, 0.9)
    j[3] = (np.nan, np.nan, 0.0)
    return PoseResult(i, t, True, j, 320, 240, (0.2, 0.1, 0.7, 0.9), None, 0.8)


def frame(i):
    return np.full((240, 320, 3), i % 255, np.uint8)


class EventTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = EventStore(self.tmp / "ev", self.tmp / "saved", max_events=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_clip(self, rec, n=150, alerts=(80,), t0=0.0):
        ids = []
        for i in range(n):
            t = t0 + i / 10
            pred = dict(probability=0.9 if i in alerts else 0.1, label=int(i in alerts), gated=False)
            eid = rec.push(t, frame(i), fake_pose(i, t, valid=i % 7 != 0), pred, new_pred=i % 5 == 0,
                           alert=i in alerts)
            if eid:
                ids.append(eid)
        rec.flush()
        rec.wait_saved()
        return ids

    def test_pre_post_window(self):
        rec = EventRecorder(self.store, pre_s=6, post_s=5, info=dict(window_size=20, model="x", threshold=0.5))
        ids = self.run_clip(rec)
        self.assertEqual(len(ids), 1)
        ev = self.store.get(ids[0])
        m, tr = ev["meta"], ev["track"]
        self.assertEqual(m["alert_frame"], 60)                    # 6 s trước @ 10 fps (+ khung báo)
        self.assertAlmostEqual(m["t_alert"], 6.0, places=3)
        self.assertEqual(len(tr), 111)                           # 6 s trước + khung báo + 5 s sau
        self.assertEqual(m["window"], [41, 60])
        self.assertEqual(m["model"], "x")
        self.assertIsNone(tr[1]["joints"])                       # clip bắt đầu ở khung 20; khung 21 (i % 7 == 0) không thấy người
        j = next(f for f in tr if f["valid"])["joints"]
        self.assertIsNone(j[3])                                  # khớp thiếu -> null
        self.assertEqual(len(list((self.store.root / ids[0] / "frames").glob("*.jpg"))), 111)
        self.assertTrue(self.store.thumb_path(ids[0]))
        self.assertFalse(list(self.store.root.glob(".tmp_*")))
        item = self.store.as_item(ids[0])                        # chạy lại được bằng model khác
        self.assertEqual(item["seq"].shape, (111, 17, 3))
        self.assertAlmostEqual(item["fps"], 10.0, places=3)

    def test_extend_and_cooldown(self):
        rec = EventRecorder(self.store, pre_s=2, post_s=3, cooldown_s=5)
        ids = self.run_clip(rec, n=200, alerts=(30, 50, 90, 150))
        # 30 mở event, 50 kéo dài (đang ghi tới 5.0+3 s), 90 trong cooldown -> bỏ, 150 mở event mới
        self.assertEqual(len(ids), 2)
        m = self.store.get(ids[0])["meta"]
        self.assertEqual(len(m["alert_frames"]), 2)
        self.assertAlmostEqual(m["duration_s"], 7.0, places=3)   # 1.0 s .. 8.0 s
        self.assertEqual(rec.skipped, 1)

    def test_prune_keeps_unreviewed(self):
        rec = EventRecorder(self.store, pre_s=1, post_s=1, cooldown_s=0)
        ids = []
        for k in range(3):
            ids += self.run_clip(rec, n=30, alerts=(15,), t0=100 * k)
            time.sleep(1.05)                                     # id theo giây
        self.store.set_label(ids[1], "false_alarm", "cúi nhặt đồ")
        ids += self.run_clip(rec, n=30, alerts=(15,), t0=1000)
        left = [m["id"] for m in self.store.list()["events"]]
        self.assertEqual(len(left), 3)
        self.assertNotIn(ids[1], left)                            # đã gắn nhãn -> xóa trước, dù không cũ nhất
        self.assertIn(ids[0], left)
        self.assertTrue(self.store.keep(ids[0]))
        d = self.store.list()
        self.assertEqual([m["id"] for m in d["saved"]], [ids[0]])
        self.assertTrue(self.store.delete(ids[2]))
        self.assertIsNone(self.store.get(ids[2]))
        self.assertIsNone(self.store.path("../../etc"))
        with self.assertRaises(ValueError):
            self.store.set_label(ids[0], "hmm")


# ------------------------------------------------------------------ MaixCAM giả lập
class FakeMaix(BaseHTTPRequestHandler):
    n_frames = 40

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            for i in range(self.n_frames):
                ok, jpg = cv2.imencode(".jpg", frame(i * 5))
                jpg = jpg.tobytes()
                joints = [[0.5, 0.1 + k * 0.04, 1.0] for k in range(17)]
                joints[5] = None
                pose = json.dumps(dict(i=i, t=i / 30, w=320, h=240, valid=i % 4 != 0, joints=joints,
                                       bbox=[0.4, 0.05, 0.6, 0.8], score=0.9, pose_ms=12.5)).encode()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                                 + b"\r\nX-Pose: " + pose + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(1 / 30)
        except OSError:
            pass


class MaixSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeMaix)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.spec = f"maix://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_url_and_json(self):
        self.assertEqual(maix_url("maix://10.0.0.5"), "http://10.0.0.5:8000/stream")
        self.assertEqual(maix_url("maix://10.0.0.5:8001"), "http://10.0.0.5:8001/stream")
        p = pose_from_json(dict(valid=True, w=640, h=480, joints=[None] + [[0.1, 0.2, 0.9]] * 16), 3, 0.3)
        self.assertTrue(p.valid and math.isnan(p.joints[0, 0]) and p.joints[0, 2] == 0)
        self.assertEqual(p.aspect, 640 / 480)
        self.assertFalse(pose_from_json(dict(valid=False), 0, 0).valid)

    def test_stream(self):
        src = MaixPoseSource(self.spec, target_fps=10)
        try:
            pk = list(src.get_frames())
        finally:
            src.release()
        self.assertGreaterEqual(len(pk), 10)                     # ~1.3 s @ 10 fps
        self.assertLessEqual(len(pk), 16)
        self.assertEqual(pk[0].frame.shape, (240, 320, 3))
        valid = [p for p in pk if p.pose.valid]
        self.assertTrue(valid)
        self.assertTrue(np.isnan(valid[0].pose.joints[5, 0]))
        self.assertEqual(valid[0].pose_ms, 12.5)
        self.assertTrue(all(b.t > a.t for a, b in zip(pk, pk[1:])))

    def test_unreachable(self):
        with self.assertRaises(RuntimeError):
            MaixPoseSource("maix://127.0.0.1:1", timeout=1)


# ------------------------------------------------------------------ web + giám sát nền (cần torch + model LSTM)
class MonitorWebTest(unittest.TestCase):
    def test_monitor_end_to_end(self):
        ck = "models/b1_lstm.pt"
        if not (ROOT / ck).is_file():
            self.skipTest("chưa có models/b1_lstm.pt")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("chưa cài torch")
        from web.config import load_config
        from web.server import create_app
        tmp = Path(tempfile.mkdtemp())
        srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeMaix)
        srv.daemon_threads = True
        FakeMaix.n_frames = 150                                   # 5 s
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            cfg = load_config(None, ["pose.backend=dummy", f"paths.output={tmp / 'out'}",
                                     f"paths.metadata={tmp / 'm.csv'}", f"paths.videos={tmp / 'v'}",
                                     f"events.dir={tmp / 'ev'}", f"events.saved_dir={tmp / 'sv'}",
                                     "events.pre_s=1", "events.post_s=1",
                                     f"classifier.checkpoint={ck}", "classifier.threshold=0.0",   # luôn báo ngã
                                     "classifier.stride_frames=2"])
            cfg["classifier"]["speed_gate"] = {"enabled": False}
            app = create_app(cfg, check_model=False)
            c = app.test_client()
            self.assertEqual(c.get("/monitor").status_code, 200)
            r = c.post("/api/monitor/start", json=dict(source=f"maix://127.0.0.1:{srv.server_address[1]}",
                                                       checkpoint=ck))
            self.assertEqual(r.status_code, 200, r.get_json())
            deadline = time.time() + 20
            while time.time() < deadline and not c.get("/api/events").get_json()["events"]:
                time.sleep(0.3)
            st = c.get("/api/monitor").get_json()
            c.post("/api/monitor/stop")
            evs = c.get("/api/events").get_json()["events"]
            self.assertTrue(evs, st)
            self.assertGreaterEqual(st["alert_seq"], 1)
            eid = evs[0]["id"]
            ev = c.get(f"/api/events/{eid}").get_json()
            self.assertEqual(ev["meta"]["model"], st["model"])
            self.assertEqual(c.get(f"/events/{eid}/frame/0.jpg").status_code, 200)
            self.assertEqual(c.get(f"/events/{eid}/thumb.jpg").status_code, 200)
            r = c.post(f"/api/events/{eid}/rerun", json=dict(checkpoint=ck)).get_json()
            self.assertTrue(r["ok"], r)
            self.assertEqual(len(r["t"]), len(r["p"]))
            self.assertEqual(c.post(f"/api/events/{eid}/rerun", json=dict(checkpoint="../secret.pt")).status_code, 400)
            self.assertTrue(c.post(f"/api/events/{eid}/label", json=dict(label="false_alarm")).get_json()["ok"])
            self.assertEqual(c.get("/api/events/..%2F..").status_code, 404)
        finally:
            FakeMaix.n_frames = 40
            srv.shutdown()
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
