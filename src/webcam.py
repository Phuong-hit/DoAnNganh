"""
webcam.py
---------
Nguồn stream (webcam / RTSP) với:
  - luồng đọc riêng, luôn chỉ giữ khung MỚI NHẤT -> xử lý chậm không bị
    dồn trễ (khung cũ bị bỏ);
  - lấy mẫu về target_fps;
  - t theo đồng hồ thật (time.monotonic) tại lúc chụp, nên đúng khi camera
    không ổn định FPS;
  - dừng được: stop(), max_frames, max_seconds, hoặc camera mất tín hiệu.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2

from .video_process import FramePacket


class StreamSource:
    def __init__(
        self,
        source=0,
        output_dir="output",
        target_fps: float = 10.0,
        max_frames: int | None = None,
        max_seconds: float | None = None,
        read_fail_limit: int = 30,
        video_id: str = "webcam",
    ):
        if target_fps <= 0:
            raise ValueError("target_fps phải > 0.")
        self.source = source
        self.video_id = video_id
        self.target_fps = float(target_fps)
        self.out_fps = self.target_fps
        self.max_frames = max_frames
        self.max_seconds = max_seconds
        self.read_fail_limit = read_fail_limit

        self.frames_dir = Path(output_dir) / "frames" / video_id  # tạo khi cần

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Không mở được nguồn stream: {source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best-effort

        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._latest = None      # (capture_time, frame)
        self._seq = 0
        self._eof = False
        self._thread = None

    # ------------------------------------------------------------------
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def _reader(self):
        fails = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails >= self.read_fail_limit:
                    break
                time.sleep(0.01)
                continue
            fails = 0
            now = time.monotonic()
            with self._cond:
                self._latest = (now, frame)
                self._seq += 1
                self._cond.notify_all()
        with self._cond:
            self._eof = True
            self._cond.notify_all()

    # ------------------------------------------------------------------
    def get_frames(self):
        """Yield FramePacket (idx, t, frame). Kết thúc khi stop()/hết giới hạn/mất camera."""
        self.start()
        period = 1.0 / self.target_fps
        k = 0
        last_seq = 0
        t0 = None
        next_emit = None

        while not self._stop.is_set():
            if self.max_frames is not None and k >= self.max_frames:
                break

            with self._cond:
                while True:
                    if self._stop.is_set():
                        return
                    if self._seq > last_seq and self._latest is not None:
                        cap_t, frame = self._latest
                        last_seq = self._seq
                        if next_emit is None or cap_t >= next_emit:
                            break
                        # chưa tới mốc lấy mẫu -> bỏ khung này
                    if self._eof:
                        return
                    self._cond.wait(timeout=0.5)

            if t0 is None:
                t0 = cap_t
                next_emit = cap_t
            if self.max_seconds is not None and cap_t - t0 > self.max_seconds:
                break

            yield FramePacket(k, round(cap_t - t0, 3), frame, -1)
            k += 1
            # Không dồn nợ: nếu xử lý chậm, mốc kế tiếp là khung mới nhất.
            next_emit = max(next_emit + period, cap_t)

    def save_frame(self, frame, frame_idx):
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        path = self.frames_dir / f"frame_{frame_idx:06d}.jpg"
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            print(f"Không thể lưu ảnh: {path}")
        return ok

    def release(self):
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# Tên cũ, giữ để code cũ vẫn chạy.
Webcam = StreamSource