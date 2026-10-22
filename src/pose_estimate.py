"""
pose_estimate.py
----------------
Ước lượng tư thế 1 người / khung hình, trả về PoseResult chuẩn hóa 0-1.

Quy ước dữ liệu (dùng chung cho cả 6 tệp):
  - joints: np.ndarray (17, 3) = [x, y, conf]
  - x, y chuẩn hóa theo (width, height) của khung gốc, nằm trong [0, 1]
  - khớp không quan sát được: x = y = NaN, conf = 0
  - valid = False nếu không có người / quá ít khớp tin cậy
    (khi đó mọi khớp là NaN) -> KHÔNG bỏ khung, vẫn trả PoseResult.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .constants import KEYPOINT_NAMES, NUM_KEYPOINTS  # noqa: F401 (giữ để import cũ vẫn chạy)


def empty_joints() -> np.ndarray:
    j = np.full((NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
    j[:, 2] = 0.0
    return j


@dataclass
class PoseResult:
    frame_idx: int
    t: float                      # giây, theo lưới target_fps (file) hoặc đồng hồ thật (stream)
    valid: bool
    joints: np.ndarray            # (17, 3) x, y (0-1), conf
    width: int = 0                # kích thước khung gốc, dùng để sửa tỉ lệ khung hình
    height: int = 0
    bbox: Optional[tuple] = None  # (x1, y1, x2, y2) chuẩn hóa 0-1
    track_id: Optional[int] = None

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.width and self.height else 1.0

    @classmethod
    def empty(cls, frame_idx, t, width=0, height=0):
        return cls(frame_idx, t, False, empty_joints(), width, height)


@dataclass
class Detection:
    """Kết quả thô của một người từ backend (đơn vị pixel)."""
    kpts_px: np.ndarray           # (17, 2)
    conf: np.ndarray              # (17,)
    bbox_px: np.ndarray           # (4,) x1, y1, x2, y2
    track_id: Optional[int] = None


class PoseBackend(Protocol):
    """Giao diện backend. Cài đặt detect() cho ONNX / MoveNet / ... là dùng được."""

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...
    def close(self) -> None: ...


class YoloBackend:
    """Backend Ultralytics. Chỉ import ultralytics khi thật sự dùng."""

    def __init__(self, model_path, conf=0.4, device=None):
        self.model_path = Path(model_path)
        # Tên trần như "yolov8n-pose.pt" (không có thư mục) thì để
        # Ultralytics tự tải về; đường dẫn có thư mục thì phải tồn tại.
        is_bare_name = self.model_path.parent == Path(".")
        if not self.model_path.exists() and not is_bare_name:
            raise FileNotFoundError(
                f"Không tìm thấy file mô hình tại: {self.model_path}"
            )
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Cần cài ultralytics để dùng YoloBackend "
                "(hoặc truyền backend khác vào PoseEstimate)."
            ) from e
        self.model = YOLO(str(self.model_path))
        self.conf = conf
        self.device = device

    def detect(self, frame_bgr):
        # predict() thay vì track(): không có trạng thái ngầm của tracker,
        # việc bám người do PoseEstimate tự làm và reset được.
        res = self.model.predict(
            source=frame_bgr, conf=self.conf, device=self.device, verbose=False
        )[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return []

        xy = res.keypoints.xy.cpu().numpy()                       # (N,17,2)
        kc = (
            res.keypoints.conf.cpu().numpy()
            if res.keypoints.conf is not None
            else np.ones(xy.shape[:2], dtype=np.float32)
        )
        boxes = (
            res.boxes.xyxy.cpu().numpy()
            if res.boxes is not None and len(res.boxes) == len(xy)
            else None
        )

        dets = []
        for i in range(len(xy)):
            if boxes is not None:
                bb = boxes[i]
            else:
                good = kc[i] > 0
                pts = xy[i][good] if good.any() else xy[i]
                bb = np.array([*pts.min(0), *pts.max(0)], dtype=np.float32)
            dets.append(Detection(xy[i], kc[i], bb))
        return dets

    def close(self):
        self.model = None


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class PoseEstimate:
    def __init__(
        self,
        model_path: str | Path | None = None,
        conf: float = 0.4,
        device: str | None = None,
        backend: PoseBackend | None = None,
        kp_conf: float = 0.3,        # ngưỡng để tính một khớp là "tin cậy"
        min_valid_joints: int = 5,   # ít hơn số này -> valid = False
        track_iou: float = 0.1,      # IoU tối thiểu để coi là cùng một người
        lost_reset_frames: int = 30, # mất người quá lâu thì quên người cũ
    ):
        if backend is None:
            if model_path is None:
                raise ValueError("Cần model_path hoặc backend.")
            backend = YoloBackend(model_path, conf=conf, device=device)
        self.backend = backend
        self.kp_conf = kp_conf
        self.min_valid_joints = min_valid_joints
        self.track_iou = track_iou
        self.lost_reset_frames = lost_reset_frames
        self._prev_bbox = None       # bbox pixel của người đang bám
        self._lost = 0

    # ------------------------------------------------------------------
    def reset(self):
        """Gọi khi bắt đầu video mới."""
        self._prev_bbox = None
        self._lost = 0

    def _select(self, dets: list[Detection]) -> Optional[Detection]:
        if not dets:
            return None
        if self._prev_bbox is not None:
            ious = [_iou(d.bbox_px, self._prev_bbox) for d in dets]
            best = int(np.argmax(ious))
            if ious[best] >= self.track_iou:
                return dets[best]
        areas = [
            max(0.0, d.bbox_px[2] - d.bbox_px[0])
            * max(0.0, d.bbox_px[3] - d.bbox_px[1])
            for d in dets
        ]
        return dets[int(np.argmax(areas))]

    def _mark_lost(self):
        self._lost += 1
        if self._lost > self.lost_reset_frames:
            self._prev_bbox = None

    # ------------------------------------------------------------------
    def process_frame(self, frame, frame_idx: int, t: float) -> PoseResult:
        h, w = frame.shape[:2]
        det = self._select(self.backend.detect(frame))

        if det is None:
            self._mark_lost()
            return PoseResult.empty(frame_idx, t, w, h)

        conf = np.nan_to_num(np.asarray(det.conf, dtype=np.float32), nan=0.0)
        xy = np.asarray(det.kpts_px, dtype=np.float32).copy()

        # Ultralytics trả (0, 0) cho khớp không có -> coi là thiếu
        missing = (conf <= 0) | ((xy[:, 0] == 0) & (xy[:, 1] == 0))
        if int(((conf >= self.kp_conf) & ~missing).sum()) < self.min_valid_joints:
            self._mark_lost()
            return PoseResult.empty(frame_idx, t, w, h)

        joints = np.empty((NUM_KEYPOINTS, 3), dtype=np.float32)
        joints[:, 0] = np.clip(xy[:, 0] / w, 0.0, 1.0)
        joints[:, 1] = np.clip(xy[:, 1] / h, 0.0, 1.0)
        joints[:, 2] = np.clip(conf, 0.0, 1.0)
        joints[missing, 0:2] = np.nan
        joints[missing, 2] = 0.0

        self._prev_bbox = det.bbox_px
        self._lost = 0
        bb = det.bbox_px
        bbox = (
            float(np.clip(bb[0] / w, 0, 1)), float(np.clip(bb[1] / h, 0, 1)),
            float(np.clip(bb[2] / w, 0, 1)), float(np.clip(bb[3] / h, 0, 1)),
        )
        return PoseResult(
            frame_idx, t, True, joints, w, h, bbox, det.track_id
        )

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()