"""Nguồn "maix://<IP>[:cổng]": nhận luồng của maixcam_app/main.py (MaixCAM chạy YOLO-pose trên NPU).

Mỗi ảnh trong luồng multipart có header X-Pose (JSON keypoint), nên PC KHÔNG chạy YOLO nữa:
gói khung trả ra có sẵn .pose (PoseResult của src/), phần còn lại của pipeline (lọc, phân loại ngã,
vẽ, lưu event) dùng y như webcam.

Giống StreamSource: luồng đọc riêng chỉ giữ khung mới nhất, lấy mẫu về target_fps, t theo đồng hồ PC.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

import cv2
import numpy as np

from src.pose_estimate import PoseResult, empty_joints

DEFAULT_PORT = 8000


def maix_url(spec: str) -> str:
    """'maix://192.168.1.50' -> 'http://192.168.1.50:8000/stream'."""
    u = urlparse(spec.strip())
    if not u.hostname:
        raise ValueError(f"Địa chỉ MaixCAM không hợp lệ: {spec} (ví dụ maix://192.168.1.50)")
    return f"http://{u.hostname}:{u.port or DEFAULT_PORT}/stream"


def pose_from_json(d: dict, frame_idx: int, t: float) -> PoseResult:
    """JSON của X-Pose -> PoseResult (khớp thiếu = NaN, conf 0 — đúng quy ước PoseEstimate)."""
    w, h = int(d.get("w") or 0), int(d.get("h") or 0)
    joints = d.get("joints")
    if not d.get("valid") or not joints:
        return PoseResult.empty(frame_idx, t, w, h)
    j = empty_joints()
    for k, p in enumerate(joints[:17]):
        if p is not None:
            j[k] = (float(p[0]), float(p[1]), float(p[2]))
    bbox = d.get("bbox")
    return PoseResult(frame_idx, t, True, j, w, h, tuple(bbox) if bbox else None, None, d.get("score"))


def read_parts(fp):
    """Đọc luồng multipart/x-mixed-replace: yield (headers: dict[str, str], jpeg: bytes).
    Cần Content-Length trong từng phần (maixcam_app/main.py luôn gửi)."""
    while True:
        line = fp.readline()
        if not line:
            return
        if not line.startswith(b"--"):
            continue
        headers = {}
        while True:
            h = fp.readline()
            if not h:
                return
            h = h.strip()
            if not h:
                break
            k, _, v = h.decode("utf-8", "replace").partition(":")
            headers[k.strip().lower()] = v.strip()
        n = int(headers.get("content-length", "0") or 0)
        if n <= 0:
            continue
        data = fp.read(n)
        if len(data) < n:
            return
        yield headers, data


@dataclass
class MaixPacket:
    idx: int
    t: float
    frame: np.ndarray
    pose: PoseResult
    pose_ms: float = 0.0          # thời gian pose đo trên MaixCAM
    src_idx: int = -1

    def __iter__(self):
        return iter((self.idx, self.t, self.frame))


class MaixPoseSource:
    def __init__(self, spec: str, target_fps: float = 10.0, timeout: float = 5.0, video_id: str = "maixcam"):
        if target_fps <= 0:
            raise ValueError("target_fps phải > 0.")
        self.spec = spec
        self.url = maix_url(spec)
        self.video_id = video_id
        self.target_fps = float(target_fps)
        self.out_fps = self.target_fps
        self.timeout = timeout
        try:
            self._resp = urllib.request.urlopen(self.url, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Không kết nối được MaixCAM {self.url}: {e}") from e
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._latest = None      # (t_nhận, jpeg, pose_dict)
        self._seq = 0
        self._eof = False
        self.error = None
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self._resp.close()                  # gỡ readline() đang chờ
        except Exception:
            pass
        with self._cond:
            self._cond.notify_all()

    def _reader(self):
        try:
            for headers, jpeg in read_parts(self._resp):
                if self._stop.is_set():
                    break
                try:
                    pose = json.loads(headers.get("x-pose") or "{}")
                except ValueError:
                    pose = {}
                with self._cond:
                    self._latest = (time.monotonic(), jpeg, pose)
                    self._seq += 1
                    self._cond.notify_all()
        except Exception as e:                    # mất mạng / MaixCAM tắt
            if not self._stop.is_set():
                self.error = str(e)
        finally:
            with self._cond:
                self._eof = True
                self._cond.notify_all()

    def get_frames(self):
        """Yield MaixPacket. Kết thúc khi stop() hoặc mất kết nối (self.error cho biết lý do)."""
        self.start()
        period = 1.0 / self.target_fps
        k, last_seq, t0, next_emit = 0, 0, None, None
        while not self._stop.is_set():
            with self._cond:
                while True:
                    if self._stop.is_set():
                        return
                    if self._seq > last_seq and self._latest is not None:
                        cap_t, jpeg, pd = self._latest
                        last_seq = self._seq
                        if next_emit is None or cap_t >= next_emit:
                            break
                    if self._eof:
                        return
                    if not self._cond.wait(timeout=self.timeout) and self._seq == last_seq:
                        self.error = f"MaixCAM không gửi khung trong {self.timeout:g} s"
                        return
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if t0 is None:
                t0 = next_emit = cap_t
            t = round(cap_t - t0, 3)
            pose = pose_from_json(pd, k, t)
            if not pose.width:                    # JSON thiếu kích thước -> lấy từ ảnh
                pose.width, pose.height = frame.shape[1], frame.shape[0]
            yield MaixPacket(k, t, frame, pose, float(pd.get("pose_ms") or 0.0))
            k += 1
            next_emit = max(next_emit + period, cap_t)

    def release(self):
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
