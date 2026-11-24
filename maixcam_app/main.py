"""MaixCAM: camera -> YOLO11-pose (NPU) -> phát ảnh + keypoint qua HTTP cho web trên PC.

MaixCAM chỉ làm luồng trực tiếp: KHÔNG chạy model ngã, KHÔNG lưu event.
PC (python run_web.py) nhận luồng, chạy LSTM / Shift-GCN, lưu 5 cảnh báo gần nhất.

Địa chỉ (IP in trên log của MaixVision và góc ảnh preview):
    http://<IP>:8000/          trang xem nhanh (mở bằng trình duyệt, để kiểm tra camera)
    http://<IP>:8000/stream    MJPEG; mỗi ảnh kèm header X-Pose = JSON keypoint
    http://<IP>:8000/info      JSON trạng thái (FPS, số người đang kéo luồng...)
Trên web PC nhập nguồn:  maix://<IP>     (cổng khác: maix://<IP>:8001)

Định dạng X-Pose (một dòng JSON):
    {"i": số khung, "t": giây (đồng hồ MaixCAM), "w": rộng, "h": cao, "valid": true/false,
     "joints": [[x, y, conf] * 17]  (x, y chuẩn hóa 0-1, khớp thiếu = null),
     "bbox": [x1, y1, x2, y2] chuẩn hóa | null, "score": conf người | null, "pose_ms": thời gian pose}
"""
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from maix import app, camera, display, image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# "Run project" của MaixVision gửi cả thư mục -> src/ nằm cạnh main.py.
# Chạy file lẻ (/tmp/maixpy_run/main.py) thì tìm ở thư mục đã chép sẵn lên máy / app đã cài.
PROJECT_DIRS = [BASE_DIR, "/root/maixcam_app", "/maixapp/apps/fall_monitor"]
MODEL_NAME = "yolo11n_pose_f11_best_v5.mud"


def _find_project():
    for d in PROJECT_DIRS:
        if os.path.isfile(os.path.join(d, "src", "pose_estimate.py")):
            return d
    return None


PROJECT_DIR = _find_project()
if PROJECT_DIR is None:
    print("\n[LỖI] Không thấy thư mục src/ cạnh main.py.\n"
          "  MaixVision đang chạy FILE LẺ (chỉ gửi main.py). Cách sửa, chọn một:\n"
          "  1) Mở cả thư mục maixcam_app (Open Project) rồi bấm 'Run Project' (không phải Run file).\n"
          "  2) Chép cả thư mục lên máy: scp -r maixcam_app root@<IP>:/root/  rồi chạy lại.\n"
          "  Đã tìm ở: " + ", ".join(PROJECT_DIRS))
    sys.exit(1)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.constants import SKELETON_EDGES  # noqa: E402
from src.pose_estimate import PoseEstimate  # noqa: E402

# Thử lần lượt: model của nhóm -> model YOLO11-pose có sẵn trong image MaixCAM (dự phòng).
# Model đầu tiên detect được một khung thật sẽ được dùng.
MODEL_CANDIDATES = [p for p in (os.path.join(PROJECT_DIR, "model_classify", MODEL_NAME),
                                os.path.join("/root/models", MODEL_NAME),
                                "/root/models/yolo11n_pose.mud")
                    if os.path.isfile(p)]
if not MODEL_CANDIDATES:
    print("\n[LỖI] Không thấy model nào: %s hoặc /root/models/yolo11n_pose.mud" % MODEL_NAME)
    sys.exit(1)

PORT = 8000
JPEG_QUALITY = 80        # 70-85 là vừa cho Wi-Fi; giảm nếu luồng bị giật
CONF_TH = 0.40
KP_CONF_TH = 0.35
SHOW_ON_SCREEN = True    # vẽ khung xương lên preview của MaixVision / màn hình (không ảnh hưởng ảnh gửi đi)


# ---------------------------------------------------------------- khung mới nhất, dùng chung cho mọi client
class Latest:
    def __init__(self):
        self.cond = threading.Condition()
        self.seq = 0
        self.jpeg = None
        self.pose = b"{}"
        self.clients = 0
        self.fps = 0.0
        self.pose_ms = 0.0

    def publish(self, jpeg, pose_json):
        with self.cond:
            self.jpeg = jpeg
            self.pose = pose_json
            self.seq += 1
            self.cond.notify_all()

    def wait(self, last_seq, timeout=2.0):
        with self.cond:
            if self.seq == last_seq:
                self.cond.wait(timeout)
            return self.seq, self.jpeg, self.pose


LATEST = Latest()


def pose_to_json(pose, idx, t, w, h, pose_ms):
    joints = None
    bbox = None
    valid = bool(pose is not None and pose.valid)
    if valid:
        joints = []
        for k in range(17):
            x, y, c = float(pose.joints[k, 0]), float(pose.joints[k, 1]), float(pose.joints[k, 2])
            if x != x or y != y or c <= 0:          # NaN -> khớp thiếu
                joints.append(None)
            else:
                joints.append([round(x, 4), round(y, 4), round(c, 3)])
        if pose.bbox is not None:
            bbox = [round(float(v), 4) for v in pose.bbox]
    d = {"i": idx, "t": round(t, 3), "w": w, "h": h, "valid": valid, "joints": joints, "bbox": bbox,
         "score": None if pose is None or pose.score is None else round(float(pose.score), 3),
         "pose_ms": round(pose_ms, 1)}
    return json.dumps(d, separators=(",", ":")).encode()


def to_jpeg_bytes(img):
    """maix.image.Image -> bytes JPEG (bộ mã hóa của MaixPy; dự phòng bằng OpenCV)."""
    try:
        return bytes(img.to_jpeg(quality=JPEG_QUALITY).to_bytes())
    except Exception:
        import cv2
        ok, buf = cv2.imencode(".jpg", image.image2cv(img, ensure_bgr=True, copy=True),
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None


# ---------------------------------------------------------------- HTTP server
PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MaixCAM</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif;text-align:center}
img{max-width:100%;max-height:85vh;margin-top:8px}</style></head><body>
<div style="padding:8px">MaixCAM pose stream &middot; web PC: nhập nguồn <b>maix://HOST</b></div>
<img src="/stream"></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            body = PAGE.replace("HOST", self.headers.get("Host", "").split(":")[0]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/info":
            body = json.dumps({"fps": round(LATEST.fps, 1), "pose_ms": round(LATEST.pose_ms, 1),
                               "clients": LATEST.clients, "frames": LATEST.seq}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/stream":
            self._stream()
        else:
            self.send_error(404)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with LATEST.cond:
            LATEST.clients += 1
        last = 0
        try:
            while not app.need_exit():
                seq, jpeg, pose = LATEST.wait(last)
                if seq == last or jpeg is None:
                    continue
                last = seq
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpeg)).encode() + b"\r\nX-Pose: " + pose + b"\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with LATEST.cond:
                LATEST.clients -= 1


def all_ips():
    """Mọi IPv4 của máy trừ 127.x: Wi-Fi (wlan0) và cáp USB (usb0 / usb_ncm...)."""
    ips = []
    try:
        for line in os.popen("ip -4 -o addr show 2>/dev/null").read().splitlines():
            parts = line.split()
            if "inet" in parts:
                ip = parts[parts.index("inet") + 1].split("/")[0]
                if not ip.startswith("127."):
                    ips.append("%s(%s)" % (ip, parts[1]))
    except Exception:
        pass
    return ips


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))        # không gửi gói nào, chỉ để hệ điều hành chọn card mạng
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        ips = all_ips()
        return ips[0].split("(")[0] if ips else "?"


# ---------------------------------------------------------------- vẽ lên màn hình (chỉ để xem trên MaixCAM)
def draw_pose(img, pose):
    if pose is None or not pose.valid:
        return
    w, h = img.width(), img.height()
    pts = []
    for k in range(17):
        x, y, c = pose.joints[k, 0], pose.joints[k, 1], pose.joints[k, 2]
        pts.append(None if c <= 0 or x != x or y != y else (int(x * w), int(y * h)))
    for a, b in SKELETON_EDGES:
        if pts[a] is not None and pts[b] is not None:
            img.draw_line(pts[a][0], pts[a][1], pts[b][0], pts[b][1], color=image.COLOR_YELLOW, thickness=2)
    for p in pts:
        if p is not None:
            img.draw_circle(p[0], p[1], 3, color=image.COLOR_RED, thickness=-1)


# ---------------------------------------------------------------- vòng chính
def open_model():
    """Nạp model + camera, chạy thử detect một khung. Lỗi -> thử model kế tiếp trong MODEL_CANDIDATES."""
    for path in MODEL_CANDIDATES:
        estimator = cam = None
        try:
            print("Loading pose model: %s" % path)
            estimator = PoseEstimate(model_path=path, conf=CONF_TH, kp_conf=KP_CONF_TH,
                                     min_valid_joints=5, dual_buff=True)
            model = estimator.backend.model
            cam = camera.Camera(model.input_width(), model.input_height(), model.input_format())
            img = None
            for _ in range(20):
                img = cam.read()
                if img is not None:
                    break
            # detect thử với ngưỡng rất thấp: luôn có ứng viên -> MaixPy phải đọc cả tensor keypoint.
            # (Ngưỡng thường mà khung không có người thì model sai output vẫn "chạy được".)
            for _ in range(3):
                model.detect(img, conf_th=0.01, iou_th=0.45, keypoint_th=0.0)
                img = cam.read() or img
            print("OK: %s (%dx%d)" % (path, model.input_width(), model.input_height()))
            return estimator, cam, path
        except Exception as e:
            print("\n[CẢNH BÁO] Model %s không chạy được: %s" % (os.path.basename(path), e))
            if "tensor idx" in str(e):
                print("  -> .cvimodel không có các output MaixPy cần (box / score / keypoint).\n"
                      "     Phải chuyển lại model, cắt ở 3 output của head (xem maixcam_app/README.md).")
            for obj in (cam, estimator):
                try:
                    if obj is not None:
                        obj.close()
                except Exception:
                    pass
    print("\n[LỖI] Không model nào chạy được.")
    sys.exit(1)


def main():
    estimator, cam, model_path = open_model()
    disp = display.Display() if SHOW_ON_SCREEN else None

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ip = local_ip()
    print("Stream: http://%s:%d/   |   web PC: maix://%s" % (ip, PORT, ip))
    print("Mọi địa chỉ của MaixCAM: %s" % (", ".join(all_ips()) or "chưa có mạng"))
    print("Model đang dùng: %s" % model_path)

    idx, fps_n, fps_t0 = 0, 0, time.time()
    try:
        while not app.need_exit():
            img = cam.read()
            if img is None:
                continue
            idx += 1
            t = time.time()
            t0 = time.perf_counter()
            try:
                pose = estimator.process_frame(frame=img, frame_idx=idx, t=t)
            except RuntimeError as e:
                if "tensor idx" not in str(e) or len(MODEL_CANDIDATES) < 2:
                    raise
                # model sai output chỉ lộ ra khi có người trong khung -> bỏ nó, chuyển model dự phòng
                print("\n[CẢNH BÁO] %s lỗi khi detect (%s) -> chuyển model dự phòng.\n"
                      "  .cvimodel thiếu output box / score / keypoint, cần chuyển lại (maixcam_app/README.md)."
                      % (os.path.basename(model_path), e))
                MODEL_CANDIDATES.remove(model_path)
                for obj in (cam, estimator):
                    try:
                        obj.close()
                    except Exception:
                        pass
                estimator, cam, model_path = open_model()
                print("Model đang dùng: %s" % model_path)
                continue
            pose_ms = (time.perf_counter() - t0) * 1000.0
            LATEST.pose_ms = pose_ms

            if LATEST.clients > 0:                  # không ai xem thì không tốn công nén JPEG
                jpeg = to_jpeg_bytes(img)          # ảnh SẠCH (chưa vẽ): PC tự vẽ khung xương
                if jpeg:
                    LATEST.publish(jpeg, pose_to_json(pose, idx, t, img.width(), img.height(), pose_ms))

            fps_n += 1
            if t - fps_t0 >= 1.0:
                LATEST.fps = fps_n / (t - fps_t0)
                fps_n, fps_t0 = 0, t
                if ip == "?":
                    ip = local_ip()

            if disp is not None:
                draw_pose(img, pose)
                img.draw_string(4, 4, "FPS %.1f  %s" % (LATEST.fps, "person" if pose.valid else "no person"),
                                color=image.COLOR_WHITE)
                img.draw_string(4, 22, "maix://%s  (%d client)" % (ip, LATEST.clients), color=image.COLOR_GREEN)
                disp.show(img)
    finally:
        server.shutdown()
        estimator.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
