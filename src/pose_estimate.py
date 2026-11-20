


import math
import numpy as np

from maix import nn


from .constants import NUM_KEYPOINTS





def empty_joints():
    joints = np.full(
        (NUM_KEYPOINTS, 3),
        np.nan,
        dtype=np.float32
    )
    joints[:, 2] = 0.0
    return joints


class PoseResult:

    def __init__(
        self,
        frame_idx,
        t,
        valid,
        joints,
        width=0,
        height=0,
        bbox=None,
        track_id=None,
        score=None
    ):
        self.frame_idx = frame_idx
        self.t = t
        self.valid = valid
        self.joints = joints
        self.width = width
        self.height = height
        self.bbox = bbox
        self.track_id = track_id
        self.score = score

    @property
    def aspect(self):
        if self.width and self.height:
            return self.width / self.height
        return 1.0

    @classmethod
    def empty(cls, frame_idx, t, width=0, height=0):
        return cls(
            frame_idx=frame_idx,
            t=t,
            valid=False,
            joints=empty_joints(),
            width=width,
            height=height
        )


class Detection:

    def __init__(
        self,
        kpts_px,
        conf,
        bbox_px,
        track_id=None,
        score=None
    ):
        self.kpts_px = kpts_px
        self.conf = conf
        self.bbox_px = bbox_px
        self.track_id = track_id
        self.score = score




class MaixPoseBackend:

    def __init__(
        self,
        model_path,
        conf=0.4,
        iou=0.5,
        kp_conf=0.35,
        dual_buff=True
    ):
        

        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.kp_conf = kp_conf

        self.model = nn.YOLO11(
            model=model_path,
            dual_buff=dual_buff
        )

    def detect(self, frame):
        

        objects = self.model.detect(
            frame,
            conf_th=self.conf,
            iou_th=self.iou,
            keypoint_th=self.kp_conf
        )

        detections = []

        for obj in objects:

            
            x1 = float(obj.x)
            y1 = float(obj.y)
            x2 = float(obj.x + obj.w)
            y2 = float(obj.y + obj.h)

            bbox = [x1, y1, x2, y2]

           
            raw_points = obj.points

            kpts = np.full(
                (NUM_KEYPOINTS, 2),
                -1,
                dtype=np.float32
            )

            point_conf = np.zeros(
                NUM_KEYPOINTS,
                dtype=np.float32
            )

            if raw_points is not None:

                
                if len(raw_points) >= NUM_KEYPOINTS * 2:

                    for k in range(NUM_KEYPOINTS):

                        px = float(raw_points[k * 2])
                        py = float(raw_points[k * 2 + 1])

                        if px >= 0 and py >= 0:
                            kpts[k, 0] = px
                            kpts[k, 1] = py

                            
                            point_conf[k] = 1.0

            
            score = getattr(obj, "score", None)

            if score is not None:
                try:
                    score = float(score)
                except Exception:
                    score = None

            detections.append(
                Detection(
                    kpts_px=kpts,
                    conf=point_conf,
                    bbox_px=np.asarray(
                        bbox,
                        dtype=np.float32
                    ),
                    track_id=None,
                    score=score
                )
            )

        return detections

    def reset(self):
        
        pass

    def close(self):
        self.model = None




def _iou(a, b):

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])

    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)

    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(
        0.0, a[3] - a[1]
    )

    area_b = max(0.0, b[2] - b[0]) * max(
        0.0, b[3] - b[1]
    )

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union




class PoseEstimate:

    def __init__(
        self,
        model_path="/root/models/yolo11n_pose_f11_best_v5.mud",
        conf=0.4,
        kp_conf=0.35,
        min_valid_joints=5,
        track_iou=0.1,
        lost_reset_frames=30,
        iou=0.5,
        use_tracker=False,
        dual_buff=True,
        backend=None
    ):

        self.backend = backend

        if self.backend is None:

            self.backend = MaixPoseBackend(
                model_path=model_path,
                conf=conf,
                iou=iou,
                kp_conf=kp_conf,
                dual_buff=dual_buff
            )

        self.kp_conf = kp_conf
        self.min_valid_joints = min_valid_joints
        self.track_iou = track_iou
        self.lost_reset_frames = lost_reset_frames

        
        self.use_tracker = False

        self._prev_bbox = None
        self._prev_id = None
        self._lost = 0

    

    def reset(self):

        self._prev_bbox = None
        self._prev_id = None
        self._lost = 0

        reset_backend = getattr(
            self.backend,
            "reset",
            None
        )

        if callable(reset_backend):
            reset_backend()

    

    def _select(self, detections):

        if not detections:
            return None

        
        if self._prev_bbox is not None:

            ious = [
                _iou(d.bbox_px, self._prev_bbox)
                for d in detections
            ]

            best_idx = int(np.argmax(ious))

            if ious[best_idx] >= self.track_iou:
                return detections[best_idx]

        
        areas = []

        for d in detections:

            bb = d.bbox_px

            area = max(
                0.0,
                bb[2] - bb[0]
            ) * max(
                0.0,
                bb[3] - bb[1]
            )

            areas.append(area)

        return detections[int(np.argmax(areas))]

    

    def _mark_lost(self):

        self._lost += 1

        if self._lost > self.lost_reset_frames:
            self._prev_bbox = None
            self._prev_id = None

    

    def process_frame(self, frame, frame_idx, t):

        
        h = frame.height()
        w = frame.width()

        detections = self.backend.detect(frame)
        det = self._select(detections)

       
        if det is None:

            self._mark_lost()

            return PoseResult.empty(
                frame_idx,
                t,
                w,
                h
            )

        conf = np.asarray(
            det.conf,
            dtype=np.float32
        )

        xy = np.asarray(
            det.kpts_px,
            dtype=np.float32
        ).copy()

        
        missing = (
            (conf <= 0)
            | (xy[:, 0] < 0)
            | (xy[:, 1] < 0)
        )

        valid_count = int(
            ((conf >= self.kp_conf) & ~missing).sum()
        )

        
        if valid_count < self.min_valid_joints:

            self._mark_lost()

            return PoseResult.empty(
                frame_idx,
                t,
                w,
                h
            )

        
        joints = np.empty(
            (NUM_KEYPOINTS, 3),
            dtype=np.float32
        )

        joints[:, 0] = np.clip(
            xy[:, 0] / w,
            0.0,
            1.0
        )

        joints[:, 1] = np.clip(
            xy[:, 1] / h,
            0.0,
            1.0
        )

        joints[:, 2] = np.clip(
            conf,
            0.0,
            1.0
        )

        # Khớp thiếu: NaN, NaN, 0
        joints[missing, 0:2] = np.nan
        joints[missing, 2] = 0.0

        # ----------------------------------------------------
        # Cập nhật tracking đơn giản bằng bbox IoU
        # ----------------------------------------------------
        self._prev_bbox = det.bbox_px.copy()
        self._prev_id = det.track_id
        self._lost = 0

        # ----------------------------------------------------
        # Bounding box normalized
        # ----------------------------------------------------
        bb = det.bbox_px

        bbox = (
            float(np.clip(bb[0] / w, 0.0, 1.0)),
            float(np.clip(bb[1] / h, 0.0, 1.0)),
            float(np.clip(bb[2] / w, 0.0, 1.0)),
            float(np.clip(bb[3] / h, 0.0, 1.0))
        )

        return PoseResult(
            frame_idx=frame_idx,
            t=t,
            valid=True,
            joints=joints,
            width=w,
            height=h,
            bbox=bbox,
            track_id=det.track_id,
            score=det.score
        )

    

    def close(self):

        if self.backend is not None:

            self.backend.close()
            self.backend = None

    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
