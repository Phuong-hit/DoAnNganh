"""Giám sát nền: chạy phát hiện ngã LIÊN TỤC, không phụ thuộc có ai mở trang web hay không.

    nguồn (maix:// | rtsp:// | webcam | file) -> pose -> lọc -> FallClassifier -> EventRecorder
                                                                         \\-> ảnh đã vẽ (MJPEG cho người xem)

- Một luồng xử lý duy nhất; bao nhiêu người xem cũng chỉ đọc "ảnh mới nhất", không mở thêm camera.
- Mất kết nối (MaixCAM tắt, Wi-Fi rớt) -> tự kết nối lại, chờ 2 s rồi tăng dần tới 15 s.
- Nguồn là file video -> chạy hết một lần rồi dừng (để thử với video ngã có sẵn).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2

from .engine import (OpenedSource, Stats, build_classifier, build_drawer, build_estimator, build_filter,
                     classifier_path, draw_fall, draw_hud, estimate_packet)
from .events import EventRecorder, EventStore


class Monitor:
    def __init__(self, cfg: dict, store: EventStore, count_frame=None):
        self.cfg = cfg
        self.store = store
        self.count_frame = count_frame or (lambda: None)
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._src: OpenedSource | None = None
        self.jpeg: bytes | None = None
        self.seq = 0
        self.alert_seq = 0                    # tăng mỗi lần chuyển sang NGÃ (web dùng để kêu chuông)
        self.opts: dict = {}
        self.state: dict = dict(running=False, status="stopped")
        self.recorder: EventRecorder | None = None

    # ------------------------------------------------------------------ điều khiển
    def start(self, source: str, checkpoint: str | None = None, use_filter: bool | None = None,
              show_skeleton: bool = True) -> None:
        source = str(source or "").strip()
        if not source:
            raise ValueError("Chưa nhập nguồn (ví dụ maix://192.168.1.50, rtsp://..., 0)")
        self.stop(wait=5.0)
        ccfg = dict(self.cfg.get("classifier") or {})
        if checkpoint:
            ccfg["checkpoint"] = checkpoint
        path = classifier_path({**self.cfg, "classifier": ccfg})
        if path is None or not Path(path).is_file():
            raise ValueError(f"Không tìm thấy model: {ccfg.get('checkpoint')}")
        ccfg["checkpoint"] = str(path)
        clf = build_classifier({**self.cfg, "classifier": ccfg})     # lỗi model báo ngay cho người bấm
        if use_filter is None:
            use_filter = bool((self.cfg.get("filter") or {}).get("enabled", True))
        self.opts = dict(source=source, checkpoint=checkpoint or (self.cfg.get("classifier") or {}).get("checkpoint"),
                         filter=bool(use_filter), skeleton=bool(show_skeleton))
        self._stop = threading.Event()
        self.state = dict(running=True, status="connecting", source=source, model=clf.name,
                          threshold=clf.fall_threshold, window_size=clf.window_size, started=time.time())
        self._thread = threading.Thread(target=self._run, args=(clf, self._stop), daemon=True, name="fall-monitor")
        self._thread.start()

    def stop(self, wait: float = 0.0) -> None:
        self._stop.set()
        src = self._src
        if src is not None and hasattr(src.src, "stop"):
            src.src.stop()
        th = self._thread
        if wait and th is not None and th is not threading.current_thread():
            th.join(wait)
        with self._cond:
            self._cond.notify_all()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        s = dict(self.state)
        s.update(running=self.running, opts=self.opts, alert_seq=self.alert_seq,
                 events_version=self.store.version,
                 recording=self.recorder.recording if self.recorder else None)
        return s

    # ------------------------------------------------------------------ người xem (MJPEG)
    def frames(self, should_stop=lambda: False):
        last = -1
        while not should_stop():
            with self._cond:
                if self.seq == last:
                    self._cond.wait(timeout=1.0)
                seq, jpeg = self.seq, self.jpeg
            if jpeg is None or seq == last:
                if not self.running:              # đã dừng: trả khung cuối (nếu chưa gửi) rồi thoát
                    return
                continue
            last = seq
            yield jpeg

    def _publish(self, img) -> None:
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._cond:
                self.jpeg = jpg.tobytes()
                self.seq += 1
                self.state["frame_at"] = time.time()
                self._cond.notify_all()

    # ------------------------------------------------------------------ vòng xử lý
    def _run(self, clf, stop: threading.Event) -> None:
        ecfg = self.cfg.get("events") or {}
        rec = EventRecorder(self.store, pre_s=float(ecfg.get("pre_s", 6)), post_s=float(ecfg.get("post_s", 5)),
                            max_clip_s=float(ecfg.get("max_clip_s", 30)), cooldown_s=float(ecfg.get("cooldown_s", 20)),
                            frame_width=int(ecfg.get("frame_width", 480)), jpeg_quality=int(ecfg.get("jpeg_quality", 75)),
                            info=dict(source=self.opts["source"], model=clf.name,
                                      checkpoint=Path(str(self.opts["checkpoint"] or "")).name or None,
                                      threshold=clf.fall_threshold, window_size=clf.window_size,
                                      stride_frames=clf.stride_frames, filter=self.opts["filter"]))
        self.recorder = rec
        estimator = None
        drawer = build_drawer(self.cfg)
        backoff = 2.0
        try:
            while not stop.is_set():
                src = None
                try:
                    self.state.update(status="connecting", error=None)
                    src = OpenedSource(self.opts["source"], self.cfg)
                    self._src = src
                    if src.kind != "maix" and estimator is None:
                        estimator = build_estimator(self.cfg)
                    flt = build_filter(self.cfg, enabled=self.opts["filter"])
                    clf.reset()
                    if estimator is not None:
                        estimator.reset()
                    self.state.update(status="live", kind=src.kind, error=None, connected_at=time.time())
                    backoff = 2.0
                    self._loop(src, estimator, flt, clf, drawer, rec, stop)
                    rec.flush()
                    err = getattr(src.src, "error", None)
                    if not src.is_live:
                        self.state.update(status="finished")
                        break
                    if not stop.is_set():
                        self.state.update(status="reconnecting", error=err or "mất tín hiệu camera")
                except Exception as e:
                    rec.flush()
                    self.state.update(status="reconnecting", error=str(e))
                finally:
                    self._src = None
                    if src is not None:
                        src.close()
                if stop.wait(backoff):
                    break
                backoff = min(backoff * 1.5, 15.0)
        finally:
            rec.flush()
            rec.wait_saved()
            if estimator is not None:
                estimator.close()
            if self.state.get("status") != "finished":
                self.state.update(status="stopped")
            self.state["running"] = False
            with self._cond:
                self._cond.notify_all()

    def _loop(self, src, estimator, flt, clf, drawer, rec, stop) -> None:
        stats = Stats()
        prev_label = 0
        hist: list = []
        wall0 = time.perf_counter()
        for pkt in src.frames():
            if stop.is_set():
                break
            if not src.is_live:                        # file: phát đúng tốc độ thật như camera
                delay = pkt.t - (time.perf_counter() - wall0)
                if delay > 0:
                    time.sleep(delay)
            pose, lat = estimate_packet(estimator, flt, pkt)
            stats.add(pose, lat)
            self.count_frame()
            res = clf.update(pose)
            alert = False
            if res is not None:
                label = int(res["label"])
                alert = label == 1 and prev_label == 0
                prev_label = label
                hist.append([round(pkt.t, 2), round(res["probability"], 4), label])
                del hist[:-200]
            if alert:
                self.alert_seq += 1
                self.state["last_alert"] = time.time()
            rec.push(pkt.t, pkt.frame, pose, clf.last, new_pred=res is not None, alert=alert)

            img = drawer.draw(pkt.frame, pose) if self.opts.get("skeleton", True) else pkt.frame.copy()
            draw_fall(img, clf.last)
            s = stats.summary()
            draw_hud(img, [f"{time.strftime('%H:%M:%S')}  {clf.name}",
                           f"{'MaixCAM ' if src.kind == 'maix' else ''}pose {lat:.0f} ms | {s['e2e_fps']:.1f} fps"])
            self._publish(img)
            last = clf.last or {}
            self.state.update(t=round(pkt.t, 2), valid=bool(pose.valid), fps=s["e2e_fps"], pose_ms=round(lat, 1),
                              valid_ratio=s["valid_ratio"], frames=s["frames"],
                              prob=None if not last else round(float(last["probability"]), 3),
                              label=None if not last else int(last["label"]),
                              gated=bool(last.get("gated", False)), collected=len(clf.buffer),
                              hist=list(hist[-100:]), skipped_alerts=rec.skipped)
