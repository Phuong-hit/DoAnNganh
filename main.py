import argparse
from datetime import datetime
from pathlib import Path
import time
from collections import deque

import cv2
import numpy as np
import torch

from src.pose_estimate import PoseEstimate
from src.skeleton_filter import SkeletonFilter
from src.skeleton_io import from_poses
from src.features import normalize_batch
from src.model_shiftgcn_4 import Model


# =========================================================
# 1. CẤU HÌNH HỆ THỐNG
# =========================================================

YOLO_MODEL_PATH = "model_yolo/yolo11n_pose_f11_best_v5.pt"
GCN_MODEL_PATH = "model_classify/shiftgcn_best_4_20_11.pt"

TARGET_FPS = 10.0          # Khớp 100% với FPS lúc tạo tập train (17 frames = 1.7s)
WINDOW_SIZE = 17
NUM_KEYPOINTS = 17
NUM_CHANNELS = 3
NUM_CLASSES = 2
MIN_VALID_RATIO = 0.50

# Thông số bộ lọc chống báo giả & giữ trạng thái cảnh báo sau ngã
FALL_THRESHOLD = 0.8       # Ngưỡng xác nhận ngã
CONSEC_FRAMES = 1          # Số frame liên tiếp vượt ngưỡng để kích hoạt
HOLD_FRAMES = 10           # Giữ cảnh báo tối thiểu 1 giây nếu chưa rõ tư thế

# Cấu hình lưu video đầu ra
SAVE_VIDEO = False
OUTPUT_DIR = Path("output/video")

# Các cặp khớp nối COCO để vẽ khung xương
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)
]


# =========================================================
# 2. BỘ LỌC HẬU XỬ LÝ & GIỮ TRẠNG THÁI SAU NGÃ
# =========================================================

class FallPostProcessor:
    """
    Hậu xử lý xác suất ngã:
    1. Kích hoạt khi raw_prob >= threshold liên tiếp consec_frames lần.
    2. Giữ cảnh báo liên tục chừng nào người chưa đứng thẳng hoàn toàn cả thân trên lẫn chân.
    3. Xả bộ đếm nhanh khi cả người đã đứng thẳng dậy.
    """

    def __init__(self, threshold=0.7, consec_frames=1, hold_frames=10):
        self.threshold = threshold
        self.consec_frames = consec_frames
        self.hold_frames = hold_frames

        self.streak = 0
        self.hold_counter = 0

    def _is_person_standing_upright(self, pose, aspect=1.0):
        """
        Kiểm tra CẢ NGƯỜI (thân trên + chân) đã đứng thẳng hoàn toàn hay chưa.
        Loại trừ trường hợp mới ngồi dậy trên sàn (vai thẳng nhưng chân gập/duỗi ngang).
        """
        if pose is None or not getattr(pose, "valid", False):
            return False

        item = from_poses([pose])
        seq = item["seq"][0]
        mask = item["mask"][0]

        valid_shoulders = [i for i in (5, 6) if mask[i]]
        valid_hips = [i for i in (11, 12) if mask[i]]
        valid_knees = [i for i in (13, 14) if mask[i]]
        valid_ankles = [i for i in (15, 16) if mask[i]]

        # Phải nhìn thấy đủ cả vai, hông và phần chân (cổ chân hoặc đầu gối)
        if not valid_shoulders or not valid_hips or (not valid_ankles and not valid_knees):
            return False

        shoulder_x = float(np.mean([seq[i, 0] for i in valid_shoulders])) * aspect
        shoulder_y = float(np.mean([seq[i, 1] for i in valid_shoulders]))

        hip_x = float(np.mean([seq[i, 0] for i in valid_hips])) * aspect
        hip_y = float(np.mean([seq[i, 1] for i in valid_hips]))

        # 1. Thân trên (Vai -> Hông) phải dựng thẳng đứng
        torso_dx = abs(shoulder_x - hip_x)
        torso_dy = hip_y - shoulder_y
        if torso_dy <= 0 or torso_dy <= torso_dx * 2.0:
            return False

        # 2. Phần chân (Hông -> Cổ chân, nếu khuất cổ chân thì dùng Gối) phải duỗi thẳng xuống dưới
        lower_indices = valid_ankles if valid_ankles else valid_knees
        lower_x = float(np.mean([seq[i, 0] for i in lower_indices])) * aspect
        lower_y = float(np.mean([seq[i, 1] for i in lower_indices]))

        leg_dx = abs(hip_x - lower_x)
        leg_dy = lower_y - hip_y
        if leg_dy <= 0 or leg_dy <= leg_dx * 2.0:
            return False

        # 3. Khi đứng thẳng, chiều dài chân (hông -> cổ chân) phải tương đương hoặc dài hơn thân trên
        # (nếu dùng đầu gối thì phải đạt ít nhất 0.65 lần chiều dài thân trên)
        min_leg_ratio = 0.85 if valid_ankles else 0.60
        if leg_dy < torso_dy * min_leg_ratio:
            return False

        return True

    def update(self, raw_prob, current_pose=None, aspect=1.0):
        if raw_prob >= self.threshold:
            self.streak += 1
        else:
            self.streak = 0

        if self.streak >= self.consec_frames:
            self.hold_counter = self.hold_frames

        if self.hold_counter > 0:
            if self._is_person_standing_upright(current_pose, aspect):
                self.hold_counter = max(0, self.hold_counter - 3)
            else:
                self.hold_counter = max(self.hold_counter, 3)
            return True

        return False

    def reset(self):
        self.streak = 0
        self.hold_counter = 0


# =========================================================
# 3. KHỞI TẠO MÔ HÌNH SHIFT-GCN
# =========================================================

def load_gcn_model(ckpt_path, device):
    model = Model(
        num_class=NUM_CLASSES,
        num_point=NUM_KEYPOINTS,
        in_channels=NUM_CHANNELS,
        num_frames=WINDOW_SIZE
    ).to(device)

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =========================================================
# 4. CHUẨN BỊ TENSOR ĐẦU VÀO & VẼ BBOX + SKELETON
# =========================================================

def build_window_tensor(pose_buffer, aspect, device):
    item = from_poses(list(pose_buffer))
    seq = item["seq"].astype(np.float32)       # [17, 17, 3]
    mask = item["mask"].astype(bool)           # [17, 17]
    valid = item["valid"].astype(bool)         # [17]

    if float(valid.mean()) < MIN_VALID_RATIO:
        return None

    X_win = seq[None, ...]
    M_win = mask[None, ...]

    X_win = normalize_batch(X_win, M_win, aspect)
    X_win = np.asarray(X_win, dtype=np.float32)

    X_win[..., 2] = M_win.astype(np.float32)
    X_win = np.transpose(X_win, (0, 3, 1, 2))

    return torch.from_numpy(np.ascontiguousarray(X_win)).to(device)


def draw_bbox_and_skeleton(frame, pose, is_fall, show_skeleton=True):
    if pose is None or not getattr(pose, "valid", False):
        return

    h, w = frame.shape[:2]

    # 1. Chỉ vẽ Bounding Box màu đỏ khi phát hiện ngã
    if is_fall:
        bbox = getattr(pose, "bbox", None)
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            pt1 = (max(0, int(round(x1))), max(0, int(round(y1))))
            pt2 = (min(w - 1, int(round(x2))), min(h - 1, int(round(y2))))
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 3)

    if not show_skeleton:
        return

    # 2. Vẽ Skeleton (hiển thị cả khớp thật conf > 0 và khớp giữ tạm của SkeletonFilter)
    joints = np.asarray(pose.joints, dtype=np.float32)
    points = []
    is_observed = []

    for i in range(NUM_KEYPOINTS):
        x, y, conf = joints[i]
        if np.isfinite(x) and np.isfinite(y) and (conf > 0 or (x > 0 and y > 0)):
            points.append((int(round(x * w)), int(round(y * h))))
            is_observed.append(conf > 0)
        else:
            points.append(None)
            is_observed.append(False)

    for start, end in SKELETON_CONNECTIONS:
        if points[start] is not None and points[end] is not None:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    for pt, obs in zip(points, is_observed):
        if pt is not None:
            # Đỏ: khớp quan sát thật | Vàng cam: khớp giữ tạm bởi SkeletonFilter
            pt_color = (0, 0, 255) if obs else (0, 200, 255)
            cv2.circle(frame, pt, 4, pt_color, -1)


# =========================================================
# 5. VÒNG LẶP CHẠY CAMERA / VIDEO THỰC TẾ
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Chạy nhận diện té ngã thời gian thực")
    parser.add_argument("--source", default="0", help="0 cho webcam hoặc đường dẫn file video")
    parser.add_argument("--yolo", default=YOLO_MODEL_PATH, help="Đường dẫn model YOLO-Pose")
    parser.add_argument("--gcn", default=GCN_MODEL_PATH, help="Đường dẫn model Shift-GCN")
    parser.add_argument("--device", default=None, help="cuda:0 hoặc cpu")
    parser.add_argument("--no-smooth", action="store_true", help="Tắt SkeletonFilter nếu muốn test raw keypoint")
    parser.add_argument("--hide-prob", action="store_true", help="Ẩn chỉ số Prob ngay khi khởi động")
    parser.add_argument("--hide-skeleton", action="store_true", help="Ẩn khung xương ngay khi khởi động")
    parser.add_argument("--save", action="store_true", help="Lưu video kết quả ra file")
    parser.add_argument("--out", default=None, help="Đường dẫn file video đầu ra (tùy chọn)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    is_webcam = args.source.isdigit()
    source = int(args.source) if is_webcam else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect = width / max(height, 1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 30.0
    frame_step = max(1, round(src_fps / TARGET_FPS))

    # Khởi tạo VideoWriter nếu bật lưu video
    writer = None
    out_path = None
    if args.save or args.out or SAVE_VIDEO:
        if args.out:
            out_path = Path(args.out)
        else:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stem = "webcam" if is_webcam else Path(args.source).stem
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"{stem}_{timestamp_str}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        save_fps = TARGET_FPS if is_webcam else src_fps
        writer = cv2.VideoWriter(str(out_path), fourcc, save_fps, (width, height))
        print(f"Đang ghi video đầu ra tại: {out_path}")

    gcn_model = load_gcn_model(args.gcn, device)
    flt = SkeletonFilter(min_conf=0.25)

    pose_buffer = deque(maxlen=WINDOW_SIZE)
    post_processor = FallPostProcessor(
        threshold=FALL_THRESHOLD,
        consec_frames=CONSEC_FRAMES,
        hold_frames=HOLD_FRAMES
    )

    empty_frames = 0
    raw_frame_idx = 0
    sample_idx = 0
    start_time = time.perf_counter()
    last_sample_time = start_time
    frame_interval = 1.0 / TARGET_FPS

    raw_prob = 0.0
    is_fall = False
    latest_pose = None
    show_prob = not args.hide_prob
    show_skeleton = not args.hide_skeleton

    print(f"Đang chạy nhận diện ở chuẩn {TARGET_FPS} FPS...")
    print("Phím tắt: 's' ẩn/hiện Skeleton | 'p' ẩn/hiện Prob | 'q' thoát.")

    try:
        with PoseEstimate(args.yolo, conf=0.25, kp_conf=0.25, imgsz=(224, 320)) as estimator:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Khóa nhịp lấy mẫu đúng 10 FPS cho cả Webcam và Video
                should_sample = False
                now = time.perf_counter()
                if is_webcam:
                    if now - last_sample_time >= frame_interval:
                        last_sample_time = now
                        should_sample = True
                else:
                    if raw_frame_idx % frame_step == 0:
                        should_sample = True

                raw_frame_idx += 1

                if should_sample:
                    # Truyền thời gian thực tế cho OneEuro khi chạy Webcam, và thời gian chuẩn 10 FPS khi chạy Video
                    t_sec = (now - start_time) if is_webcam else (sample_idx / TARGET_FPS)
                    pose = estimator.process_frame(frame, sample_idx, t_sec)
                    if not args.no_smooth:
                        pose = flt.smooth(pose)
                    latest_pose = pose
                    sample_idx += 1

                    if not pose.valid:
                        empty_frames += 1
                        if empty_frames >= 5:
                            pose_buffer.clear()
                            post_processor.reset()
                            flt.reset()
                    else:
                        empty_frames = 0

                    pose_buffer.append(pose)

                    if len(pose_buffer) == WINDOW_SIZE:
                        tensor_in = build_window_tensor(pose_buffer, aspect, device)

                        if tensor_in is not None:
                            with torch.no_grad():
                                logits = gcn_model(tensor_in)
                                probs = torch.softmax(logits, dim=1)
                                raw_prob = float(probs[0, 1].item())

                        is_fall = post_processor.update(raw_prob, pose, aspect)

                # Chỉ vẽ Bounding Box màu đỏ khi bị ngã, kèm Skeleton nếu đang bật
                draw_bbox_and_skeleton(frame, latest_pose, is_fall, show_skeleton=show_skeleton)

                # Hiển thị chữ trạng thái NGA / KHONG NGA
                if is_fall:
                    text_color = (0, 0, 255)
                    status_text = f"NGA! (Prob: {raw_prob:.0%})" if show_prob else "NGA!"
                else:
                    text_color = (0, 255, 0)
                    status_text = f"KHONG NGA (Prob: {raw_prob:.0%})" if show_prob else "KHONG NGA"

                cv2.putText(
                    frame,
                    status_text,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    text_color,
                    2,
                    cv2.LINE_AA
                )

                if writer is not None:
                    writer.write(frame)

                cv2.imshow("Fall Detection - Shift-GCN", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    show_prob = not show_prob
                elif key == ord("s"):
                    show_skeleton = not show_skeleton
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Đã lưu video thành công: {out_path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()