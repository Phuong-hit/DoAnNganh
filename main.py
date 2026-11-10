
"""
main.py
-------
Phát hiện ngã từ video bằng YOLO Pose + Shift-GCN.

Pipeline:
    Video
      -> YOLO11n-Pose
      -> 17 COCO keypoints
      -> Shift-GCN
      -> ADL / FALL

Nhãn:
    0 = ADL (Không ngã)
    1 = FALL (Ngã)
"""

from pathlib import Path
from dataclasses import dataclass
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from ultralytics import YOLO

from src.shiftgcn_inference import ShiftGCNInference


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

POSE_MODEL_PATH = ROOT / "model_yolo" / "yolo11n_pose_f11_best_v5.pt"
CHECKPOINT_PATH = ROOT / "model_classify" / "shiftgcn_bestv5.pt"

POSE_CONF = 0.25

# Tốc độ lấy mẫu video đưa vào bộ phân loại.
# Phải phù hợp với tốc độ lấy mẫu khi tạo dataset huấn luyện.
TARGET_FPS = 10.0

WINDOW_SIZE = 17
STRIDE_FRAMES = 1

FALL_THRESHOLD = 0.5

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 450

SAVE_VIDEO = False

OUTPUT_DIR = ROOT / "output" / "video"


# ============================================================
# 2. COCO KEYPOINTS AND SKELETON
# ============================================================

COCO_KEYPOINTS = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle",
]

SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


# ============================================================
# 3. POSE RESULT
# ============================================================

@dataclass
class PoseResult:
    joints: np.ndarray
    valid: bool
    t: float
    frame_idx: int
    aspect: float
    width: int
    height: int


# ============================================================
# 4. SELECT VIDEO
# ============================================================

def select_video():

    root = tk.Tk()
    root.withdraw()

    video_path = filedialog.askopenfilename(
        title="Chọn video cần phát hiện ngã",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    return video_path


def select_input_mode():
    print("\n========== CHỌN NGUỒN ĐẦU VÀO ==========")
    print("1. Video có sẵn")
    print("2. Webcam trực tiếp")

    while True:
        choice = input("Chọn chế độ (1/2): ").strip()

        if choice in ("1", "2"):
            return choice

        print("Lựa chọn không hợp lệ!")

# ============================================================
# 5. EXTRACT POSE
# ============================================================

def extract_pose(result, frame, frame_idx, timestamp):

    height, width = frame.shape[:2]

    aspect = width / height if height > 0 else 1.0

    joints = np.zeros((17, 3), dtype=np.float32)

    if (
        result.keypoints is None
        or result.boxes is None
        or len(result.boxes) == 0
    ):
        return PoseResult(
            joints=joints,
            valid=False,
            t=timestamp,
            frame_idx=frame_idx,
            aspect=aspect,
            width=width,
            height=height,
        )

    boxes = result.boxes.xyxy.cpu().numpy()

    # Chọn người có bounding box lớn nhất.
    areas = (
        (boxes[:, 2] - boxes[:, 0])
        * (boxes[:, 3] - boxes[:, 1])
    )

    best_idx = int(np.argmax(areas))

    keypoints = result.keypoints.data.cpu().numpy()

    if best_idx >= len(keypoints):
        return PoseResult(
            joints=joints,
            valid=False,
            t=timestamp,
            frame_idx=frame_idx,
            aspect=aspect,
            width=width,
            height=height,
        )

    points = keypoints[best_idx]

    if points.shape[0] != 17:
        raise ValueError(
            f"YOLO trả về {points.shape[0]} keypoints, "
            "nhưng chương trình yêu cầu 17 keypoints."
        )

    # Chuẩn hóa tọa độ về [0, 1].
    joints[:, 0] = points[:, 0] / width
    joints[:, 1] = points[:, 1] / height

    if points.shape[1] >= 3:
        joints[:, 2] = points[:, 2]

    valid = bool(np.any(joints[:, 2] > 0))

    return PoseResult(
        joints=joints,
        valid=valid,
        t=timestamp,
        frame_idx=frame_idx,
        aspect=aspect,
        width=width,
        height=height,
    )


# ============================================================
# 6. DRAW SKELETON
# ============================================================

def draw_skeleton(frame, pose):

    if pose is None or not pose.valid:
        return frame

    joints = pose.joints

    height, width = frame.shape[:2]

    points = []

    for x, y, conf in joints:

        if conf <= 0:
            points.append(None)
            continue

        px = int(x * width)
        py = int(y * height)

        points.append((px, py))

    # Vẽ các xương.
    for start, end in SKELETON:

        p1 = points[start]
        p2 = points[end]

        if p1 is None or p2 is None:
            continue

        cv2.line(
            frame,
            p1,
            p2,
            (0, 255, 0),
            2,
        )

    # Vẽ keypoints.
    for point in points:

        if point is None:
            continue

        cv2.circle(
            frame,
            point,
            4,
            (0, 0, 255),
            -1,
        )

    return frame


# ============================================================
# 7. RESIZE FRAME
# ============================================================

def resize_frame(frame):

    return cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# 8. DRAW PREDICTION
# ============================================================

def draw_prediction(frame, prediction):

    if prediction is None:
        text = "Waiting for 17 frames..."
        color = (255, 255, 255)

    else:

        label = prediction["label"]
        probability = prediction["probability"]

        if label == 1:
            text = f"FALL - {probability * 100:.1f}%"
            color = (0, 0, 255)

        else:
            text = f"ADL - {probability * 100:.1f}%"
            color = (0, 255, 0)

    cv2.rectangle(
        frame,
        (10, 10),
        (350, 55),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    return frame


# ============================================================
# 9. MAIN
# ============================================================


def main():

    print("=" * 55)
    print("       SHIFT-GCN FALL DETECTION")
    print("=" * 55)

    if not POSE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy YOLO Pose:\n{POSE_MODEL_PATH}"
        )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy Shift-GCN checkpoint:\n{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Chọn chế độ đầu vào
    # --------------------------------------------------------

    mode = select_input_mode()

    if mode == "1":
        video_path = select_video()

        if not video_path:
            print("Chưa chọn video.")
            return

        cap = cv2.VideoCapture(video_path)
        window_title = "Shift-GCN - Video"

    else:
        # 0 là webcam mặc định.
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        window_title = "Shift-GCN - Webcam"

    if not cap.isOpened():
        raise RuntimeError(
            "Không thể mở nguồn video/webcam."
        )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    if mode == "1":
        source_fps = cap.get(cv2.CAP_PROP_FPS)

        if source_fps <= 0:
            source_fps = 30.0

    else:
        source_fps = TARGET_FPS

    total_frames = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if mode == "1" else 0
    )

    print(f"\nFPS lấy mẫu: {TARGET_FPS}")
    print(f"Cửa sổ: {WINDOW_SIZE} frame")
    print(f"Stride: {STRIDE_FRAMES} frame")

    # --------------------------------------------------------
    # Load YOLO Pose
    # --------------------------------------------------------

    print("\nĐang tải YOLO Pose...")

    pose_model = YOLO(str(POSE_MODEL_PATH))

    print("Đã tải YOLO Pose.")

    # --------------------------------------------------------
    # Load Shift-GCN
    # --------------------------------------------------------

    print("\nĐang tải Shift-GCN...")

    classifier = ShiftGCNInference(
        checkpoint_path=CHECKPOINT_PATH,
        window_size=WINDOW_SIZE,
        stride_frames=STRIDE_FRAMES,
        fall_threshold=FALL_THRESHOLD,
    )

    print("Đã tải Shift-GCN.")

    # --------------------------------------------------------
    # Video sampling
    # --------------------------------------------------------

    sample_interval = 1.0 / TARGET_FPS

    next_sample_time = 0.0
    frame_idx = 0

    latest_prediction = None
    latest_pose = None

    # Webcam sử dụng thời gian thực.
    import time
    start_time = time.monotonic()

    print("\nĐang chạy. Nhấn Q để thoát.\n")

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if mode == "1":
            timestamp = frame_idx / source_fps
        else:
            timestamp = time.monotonic() - start_time

        # Chỉ lấy mẫu khi đến thời điểm cần xử lý.
        if timestamp >= next_sample_time:

            results = pose_model.predict(
                frame,
                conf=POSE_CONF,
                verbose=False,
            )

            result = results[0]

            pose = extract_pose(
                result=result,
                frame=frame,
                frame_idx=frame_idx,
                timestamp=timestamp,
            )

            latest_pose = pose

            prediction = classifier.update(pose)

            if prediction is not None:
                latest_prediction = prediction

            # Tránh xử lý bù nhiều frame nếu YOLO chạy chậm.
            next_sample_time = timestamp + sample_interval

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        display_frame = frame.copy()

        display_frame = draw_skeleton(
            display_frame,
            latest_pose,
        )

        display_frame = draw_prediction(
            display_frame,
            latest_prediction,
        )

        # Hiển thị FPS thực tế.
        if mode == "2":
            elapsed = time.monotonic() - start_time
            current_fps = frame_idx / elapsed if elapsed > 0 else 0

            cv2.putText(
                display_frame,
                f"FPS: {current_fps:.1f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        display_frame = resize_frame(display_frame)

        cv2.imshow(
            window_title,
            display_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        frame_idx += 1

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()
    cv2.destroyAllWindows()

    print("\nĐã kết thúc chương trình.")


if __name__ == "__main__":
    main()
