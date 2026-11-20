import os
import time

from maix import app, camera, display, image

from src.pose_estimate import PoseEstimate
from src.constants import SKELETON_EDGES




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "model_classify", "yolo11n_pose_f11_best_v5.mud"
)

CONF_TH = 0.40
KP_CONF_TH = 0.35

SHOW_FPS = True
SHOW_BBOX = True
SHOW_KEYPOINTS = True
SHOW_SKELETON = True




def draw_pose(img, pose):
    if pose is None or not pose.valid:
        return

    w_img = img.width()
    h_img = img.height()
    joints = pose.joints

    # bbox normalized (x1, y1, x2, y2)
    if SHOW_BBOX and pose.bbox is not None:
        x1, y1, x2, y2 = pose.bbox
        px1, py1 = int(x1 * w_img), int(y1 * h_img)
        px2, py2 = int(x2 * w_img), int(y2 * h_img)
        img.draw_rect(
            px1, py1, px2 - px1, py2 - py1,
            color=image.COLOR_GREEN, thickness=2
        )

    # keypoints -> pixel
    points = []
    for i in range(17):
        x, y, c = joints[i, 0], joints[i, 1], joints[i, 2]
        if c <= 0 or x != x or y != y:  
            points.append(None)
        else:
            points.append((int(x * w_img), int(y * h_img)))

    if SHOW_SKELETON:
        for a, b in SKELETON_EDGES:
            p1, p2 = points[a], points[b]
            if p1 is None or p2 is None:
                continue
            img.draw_line(
                p1[0], p1[1], p2[0], p2[1],
                color=image.COLOR_YELLOW, thickness=2
            )

    if SHOW_KEYPOINTS:
        for p in points:
            if p is None:
                continue
            img.draw_circle(
                p[0], p[1], 3,
                color=image.COLOR_RED, thickness=-1
            )



print("Loading model...")
pose_estimator = PoseEstimate(
    model_path=MODEL_PATH,
    conf=CONF_TH,
    kp_conf=KP_CONF_TH,
    min_valid_joints=5,
    dual_buff=True
)
print("Model loaded!")


model = pose_estimator.backend.model
cam = camera.Camera(
    model.input_width(),
    model.input_height(),
    model.input_format()
)
disp = display.Display()




frame_idx = 0
fps = 0.0
fps_count = 0
fps_start = time.time()

while not app.need_exit():
    img = cam.read()
    if img is None:
        continue

    frame_idx += 1
    now = time.time()

    pose = pose_estimator.process_frame(
        frame=img, frame_idx=frame_idx, t=now
    )

    if pose is not None and pose.valid:
        draw_pose(img, pose)
        img.draw_string(10, 35, "Person detected",
                        color=image.COLOR_GREEN)
    else:
        img.draw_string(10, 35, "No person detected",
                        color=image.COLOR_RED)

    fps_count += 1
    elapsed = now - fps_start
    if elapsed >= 1.0:
        fps = fps_count / elapsed
        fps_count = 0
        fps_start = now

    if SHOW_FPS:
        img.draw_string(10, 10, "FPS: {:.1f}".format(fps),
                        color=image.COLOR_WHITE)

    disp.show(img)


pose_estimator.close()
print("Stopped.")