
"""
shiftgcn_inference.py
---------------------
Suy luận phát hiện ngã từ chuỗi PoseResult bằng Shift-GCN.

Nhãn:
    0 = KHÔNG NGÃ (ADL)
    1 = NGÃ (Fall)

Mỗi lần dự đoán sử dụng 17 frame.
Mặc định trượt 1 frame giữa hai lần dự đoán.

Đầu vào:
    PoseResult của từng frame.

Đầu vào Shift-GCN:
    [N, C, T, V] = [1, 3, 17, 17]

Checkpoint:
    shiftgcn_best.pt được lưu bởi train.py Shift-GCN.
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from .features import normalize_clip
from .skeleton_io import from_poses
from .model_shiftgcn import Model


class ShiftGCNInference:

    def __init__(
        self,
        checkpoint_path,
        window_size=17,
        stride_frames=1,
        device=None,
        fall_threshold=0.5,
    ):

        # =================================================
        # 1. CONFIGURATION
        # =================================================

        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy trọng số: {self.checkpoint_path}"
            )

        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.window_size = int(window_size)
        self.stride_frames = int(stride_frames)
        self.fall_threshold = float(fall_threshold)

        if self.window_size < 1:
            raise ValueError("window_size phải >= 1.")

        if self.stride_frames < 1:
            raise ValueError("stride_frames phải >= 1.")

        if not 0 <= self.fall_threshold <= 1:
            raise ValueError("fall_threshold phải nằm trong [0, 1].")

        # =================================================
        # 2. LOAD CHECKPOINT
        # =================================================

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )

        except TypeError:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )

        # Checkpoint phải được lưu theo định dạng train.py.
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint không đúng định dạng.")

        required_keys = [
            "model_state_dict",
            "num_frames",
            "num_point",
            "in_channels",
            "num_class",
        ]

        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(
                    f"Checkpoint thiếu thông tin: {key}"
                )

        # =================================================
        # 3. CHECK MODEL CONFIGURATION
        # =================================================

        num_frames = int(checkpoint["num_frames"])
        num_point = int(checkpoint["num_point"])
        in_channels = int(checkpoint["in_channels"])
        num_class = int(checkpoint["num_class"])

        if self.window_size != num_frames:
            raise ValueError(
                f"Checkpoint yêu cầu {num_frames} frame, "
                f"nhưng window_size đang là {self.window_size}."
            )

        if num_point != 17:
            raise ValueError(
                f"Checkpoint có {num_point} keypoint, "
                "không đúng COCO 17 keypoint."
            )

        if in_channels != 3:
            raise ValueError(
                f"Checkpoint yêu cầu {in_channels} kênh đầu vào."
            )

        if num_class != 2:
            raise ValueError(
                f"Checkpoint có {num_class} lớp đầu ra, "
                "không đúng bài toán ADL/Fall."
            )

        # =================================================
        # 4. INITIALIZE SHIFT-GCN
        # =================================================

        self.model = Model(
            num_class=num_class,
            num_point=num_point,
            in_channels=in_channels,
            num_frames=num_frames,
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.eval()

        # =================================================
        # 5. SLIDING WINDOW BUFFER
        # =================================================

        self.buffer = deque(
            maxlen=self.window_size
        )

        self.frames_since_prediction = 0

        print("=" * 55)
        print("SHIFT-GCN INFERENCE")
        print("=" * 55)

        print("Checkpoint:", self.checkpoint_path)
        print("Device:", self.device)
        print("Window size:", self.window_size)
        print("Stride frames:", self.stride_frames)
        print("Fall threshold:", self.fall_threshold)

        print("Shift-GCN loaded successfully.")

    # =====================================================
    # 6. RESET
    # =====================================================

    def reset(self):
        """
        Xóa lịch sử khi chuyển video,
        chuyển người hoặc bắt đầu phiên mới.
        """

        self.buffer.clear()
        self.frames_since_prediction = 0

    # =====================================================
    # 7. PREPARE INPUT
    # =====================================================

    def prepare_input(self, poses):

        # Chuyển PoseResult thành:
        # seq:  [T,17,3]
        # mask: [T,17]

        item = from_poses(poses)

        # Chuẩn hóa giống pipeline huấn luyện.
        clip = normalize_clip(
            item["seq"],
            item["mask"],
            aspect=item.get("aspect", 1.0),
        )

        clip = np.asarray(
            clip,
            dtype=np.float32,
        )

        expected_shape = (
            self.window_size,
            17,
            3,
        )

        if clip.shape != expected_shape:
            raise ValueError(
                f"Shape sau chuẩn hóa không đúng: {clip.shape}. "
                f"Yêu cầu: {expected_shape}"
            )

        if not np.isfinite(clip).all():
            raise ValueError(
                "Dữ liệu khung xương có NaN hoặc Inf."
            )

        # [T,17,3] -> [1,3,T,17]
        X = clip.transpose(2, 0, 1)[np.newaxis, ...]

        X = np.ascontiguousarray(
            X,
            dtype=np.float32,
        )

        return torch.from_numpy(X).to(self.device)

    # =====================================================
    # 8. PREDICTION
    # =====================================================

    @torch.inference_mode()
    def predict(self, poses):

        x = self.prepare_input(poses)

        # Shift-GCN forward.
        logits = self.model(x)

        # Chuyển logits thành xác suất.
        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        p_no_fall = float(
            probabilities[0].item()
        )

        p_fall = float(
            probabilities[1].item()
        )

        # Phân loại theo ngưỡng.
        label = int(
            p_fall >= self.fall_threshold
        )

        return {
            "label": label,

            "name": (
                "NGÃ"
                if label == 1
                else "KHÔNG NGÃ"
            ),

            "probability": p_fall,

            "probability_percent": p_fall * 100.0,

            "probabilities": {
                "khong_nga": p_no_fall,
                "nga": p_fall,
            },

            "percentages": {
                "khong_nga": p_no_fall * 100.0,
                "nga": p_fall * 100.0,
            },
        }

    # =====================================================
    # 9. UPDATE - SLIDING WINDOW
    # =====================================================

    def update(self, pose):
        """
        Nhận một PoseResult của frame hiện tại.

        Trả về None:
            - Khi chưa đủ 17 frame.
            - Khi chưa đến thời điểm dự đoán tiếp theo.

        Trả về dict:
            - Khi đã đủ cửa sổ và đến lượt dự đoán.
        """

        self.buffer.append(pose)

        self.frames_since_prediction += 1

        # Chưa đủ 17 frame.
        if len(self.buffer) < self.window_size:
            return None

        # Chưa đến lượt dự đoán tiếp theo.
        if (
            self.frames_since_prediction
            < self.stride_frames
        ):
            return None

        self.frames_since_prediction = 0

        # Dự đoán trên cửa sổ hiện tại.
        return self.predict(
            list(self.buffer)
        )
