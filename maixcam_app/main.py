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
from collections import deque
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
from src.skeleton_filter import SkeletonFilter  # noqa: E402
from src.fall_detect import FallDetector  # noqa: E402

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
CONF_TH = 0.25           # giống web / nhánh final (trước: 0.40)
KP_CONF_TH = 0.25        # giống web / nhánh final (trước: 0.35)
ENABLE_FALL = True       # chạy phát hiện ngã NGAY TRÊN MaixCAM (Shift-GCN numpy, model_classify/*.npz)
CPU_BENCH_ON_START = True  # đo tốc độ CPU cho Shift-GCN 1 lần lúc khởi động (~5-15 s), kết quả ở log + /bench
FALL_USE_NPU = True      # có shiftgcn_npu.mud/.cvimodel thì chạy Shift-GCN trên NPU (lỗi -> tự về numpy CPU)
FALL_SAMPLE_FPS = 10.0  # lấy mẫu 10 khung/s cho bộ phân loại (cửa sổ 17 khung = 1.7 s, như lúc train)
FILTER_MIN_CONF = 0.25   # SkeletonFilter(min_conf) như nhánh final
SHOW_ON_SCREEN = True    # hiện ảnh camera + "NGA! / KHONG NGA  P=..." lên màn hình / preview MaixVision
SHOW_SKELETON = False    # True: vẽ thêm khung xương, FPS, IP lên màn hình (không ảnh hưởng ảnh gửi lên web)


# ---------------------------------------------------------------- đo hiệu năng (đề cương 3.3): thời gian + RAM
def _proc_status_kb(key):
    """VmHWM = peak RSS của tiến trình, VmRSS = RSS hiện tại (KB)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except Exception:
        pass
    return None


def _meminfo_kb():
    m = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                m[k] = int(v.split()[0])
    except Exception:
        pass
    return m


def mem_used_mb():
    """RAM cả hệ thống đang dùng = MemTotal - MemAvailable (gồm cả bộ nhớ ngoài tiến trình Python)."""
    m = _meminfo_kb()
    if "MemTotal" in m and "MemAvailable" in m:
        return (m["MemTotal"] - m["MemAvailable"]) / 1024.0
    return None


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1))))]


class Bench:
    """Thời gian từng bước mỗi khung (ms) + RAM. Giữ BENCH_KEEP khung gần nhất. GET /bench để lấy, /bench/reset để đo lại."""
    # fall_ms = lọc + cửa sổ + model + hậu xử lý; model_ms = riêng Shift-GCN; e2e_ms = đọc camera -> quyết định.
    # 3 bước này chỉ có ở khung được lấy mẫu cho bộ phân loại (10 khung/s).
    # fall_ms = filter_ms (SkeletonFilter) + prep_ms (cửa sổ + chuẩn hóa) + model_ms + post_ms (hậu xử lý)
    STEPS = ("read_ms", "pose_ms", "fall_ms", "filter_ms", "prep_ms", "model_ms", "post_ms", "e2e_ms",
             "encode_ms", "display_ms", "loop_ms")

    def __init__(self, keep=3000):
        self.keep = keep
        self.lock = threading.Lock()
        self.mem_before_model = mem_used_mb()     # gọi TRƯỚC khi nạp model
        self.mem_total = (_meminfo_kb().get("MemTotal") or 0) / 1024.0
        self.reset()

    def reset(self):
        with self.lock:
            self.data = {k: [] for k in self.STEPS}
            self.frames = 0
            self.samples = 0
            self.sample_times = deque(maxlen=400)   # thời điểm lấy mẫu gần đây (tính tần số 10 s gần nhất)
            self.alerts = 0
            self.persons = 0
            self.t0 = time.time()
            self.mem_peak = mem_used_mb() or 0.0

    def add_fall(self, **ms):
        """Thời gian bước phát hiện ngã (chỉ khung được lấy mẫu). Giá trị None thì bỏ qua."""
        with self.lock:
            for k, v in ms.items():
                if v is None:
                    continue
                lst = self.data[k]
                lst.append(v)
                if len(lst) > self.keep:
                    del lst[0]
            self.samples += 1
            self.sample_times.append(time.time())

    def add(self, valid, **ms):
        with self.lock:
            for k, v in ms.items():
                lst = self.data[k]
                lst.append(v)
                if len(lst) > self.keep:
                    del lst[0]
            self.frames += 1
            self.persons += int(bool(valid))
            if self.frames % 10 == 0:              # đọc /proc 1 lần / 10 khung cho nhẹ
                used = mem_used_mb()
                if used is not None and used > self.mem_peak:
                    self.mem_peak = used

    def summary(self):
        with self.lock:
            out = {"frames": self.frames, "samples": self.samples, "alerts": self.alerts,
                   "since_s": round(time.time() - self.t0, 1),
                   "person_ratio": round(self.persons / self.frames, 3) if self.frames else None}
            for k, xs in self.data.items():
                if xs:
                    out[k + "_mean"] = round(sum(xs) / len(xs), 2)
                    out[k + "_p50"] = round(_pct(xs, 50), 2)
                    out[k + "_p95"] = round(_pct(xs, 95), 2)
                    out[k + "_max"] = round(max(xs), 2)
            loop = out.get("loop_ms_mean")
            out["fps"] = round(1000.0 / loop, 2) if loop else None
            # Tần số đưa khung vào bộ phân loại (model train ở 10/s). Thấp hơn -> cửa sổ 17 khung dài hơn 1.7 s,
            # chuyển động chậm trông nhanh hơn -> dễ báo nhầm (vd. nằm xuống chậm).
            el = time.time() - self.t0
            out["sample_hz_total"] = round(self.samples / el, 2) if el > 0 and self.samples else None
            now = time.time()
            recent = [x for x in self.sample_times if now - x <= 10.0]
            span = min(10.0, now - self.t0)
            out["sample_hz"] = round(len(recent) / span, 2) if span > 1.0 and recent else None   # 10 s gần nhất
            hwm, rss = _proc_status_kb("VmHWM"), _proc_status_kb("VmRSS")
            out["vm_hwm_mb"] = round(hwm / 1024.0, 1) if hwm else None       # peak RAM tiến trình
            out["vm_rss_mb"] = round(rss / 1024.0, 1) if rss else None
            out["mem_total_mb"] = round(self.mem_total, 1)
            out["mem_used_before_mb"] = None if self.mem_before_model is None else round(self.mem_before_model, 1)
            out["mem_used_now_mb"] = None if mem_used_mb() is None else round(mem_used_mb(), 1)
            out["mem_used_peak_mb"] = round(self.mem_peak, 1)                 # peak RAM cả hệ thống khi chạy
            if self.mem_before_model is not None:
                out["mem_pipeline_mb"] = round(self.mem_peak - self.mem_before_model, 1)
            return out


BENCH = None   # tạo trong main(), trước khi nạp model


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


def pose_to_json(pose, idx, t, w, h, pose_ms, fall=None):
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
    if fall is not None:                            # kết quả phát hiện ngã của chính MaixCAM
        d["fall"] = {"p": round(float(fall["prob"]), 4), "label": int(fall["label"]), "held": bool(fall["held"]),
                     "gated": bool(fall.get("gated"))}
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
        elif path in ("/bench", "/bench/reset"):
            if path == "/bench/reset" and BENCH is not None:
                BENCH.reset()
            data = BENCH.summary() if BENCH is not None else {}
            data["model"] = os.path.basename(CURRENT_MODEL[0] or "")
            data.update(MODEL_INFO)
            data["clients"] = LATEST.clients
            body = json.dumps(data).encode()
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


CURRENT_MODEL = [None]
MODEL_INFO = {}


def model_info(mud_path, estimator):
    """Kích thước đầu vào + dung lượng file .cvimodel (file .mud chỉ trỏ tới nó)."""
    info = {}
    try:
        m = estimator.backend.model
        info["input"] = [m.input_width(), m.input_height()]
    except Exception:
        pass
    try:
        cvi = None
        with open(mud_path) as f:
            for line in f:
                if line.strip().startswith("model") and "=" in line and line.split("=")[0].strip() == "model":
                    cvi = os.path.join(os.path.dirname(mud_path), line.split("=", 1)[1].strip())
        if cvi and os.path.isfile(cvi):
            info["model_file"] = os.path.basename(cvi)
            info["model_file_mb"] = round(os.path.getsize(cvi) / 2.0 ** 20, 2)
    except Exception:
        pass
    return info


def open_fall_detector():
    """Shift-GCN (numpy) + SkeletonFilter cho phát hiện ngã trên board. Lỗi -> (None, None, {}): vẫn phát luồng pose."""
    if not ENABLE_FALL:
        return None, None, {}
    d = os.path.join(PROJECT_DIR, "model_classify")
    files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and not f.endswith("_check.npz")) \
        if os.path.isdir(d) else []
    if not files:
        print("\n[CẢNH BÁO] Không thấy model phát hiện ngã (.npz) trong %s -> chỉ chạy pose.\n"
              "  Trên PC: python export_maixcam_model.py rồi Run Project lại." % d)
        return None, None, {}
    path = os.path.join(d, files[0])
    mud = os.path.join(d, "shiftgcn_npu.mud")
    check = os.path.join(d, "shiftgcn_npu_check.npz")
    use_npu = FALL_USE_NPU and os.path.isfile(mud) and os.path.isfile(os.path.join(d, "shiftgcn_npu.cvimodel"))
    try:
        det = FallDetector.load(path, npu_mud=mud if use_npu else None,
                                check_npz=check if os.path.isfile(check) else None)
        flt = SkeletonFilter(min_conf=FILTER_MIN_CONF)
    except Exception as e:
        print("\n[CẢNH BÁO] Không nạp được %s: %s -> chỉ chạy pose." % (files[0], e))
        return None, None, {}
    li = getattr(det, "load_info", {})
    backend = li.get("fall_backend", "numpy")
    model_file = "shiftgcn_npu.cvimodel" if backend == "npu" else files[0]
    info = {"fall_model": model_file,
            "fall_model_kb": round(os.path.getsize(os.path.join(d, model_file)) / 1024.0, 1),
            "fall_params": det.net.num_params, "fall_threshold": det.threshold,
            "fall_window": det.window_size, "fall_sample_fps": FALL_SAMPLE_FPS}
    info.update(li)
    if use_npu and backend != "npu":
        print("\n[CẢNH BÁO] Shift-GCN trên NPU không dùng được (%s) -> chạy numpy trên CPU (chậm)."
              % li.get("fall_npu_error"))
    elif backend == "npu":
        print("Shift-GCN trên NPU: tự kiểm tra lệch %s, đổi nhãn %s (so với PyTorch trên 8 clip thật)"
              % (li.get("fall_npu_check_err"), li.get("fall_npu_check_flips")))
    print("Phát hiện ngã trên MaixCAM: %s [%s] (%d tham số, ngưỡng %.2f, cửa sổ %d khung @ %g FPS)"
          % (model_file, backend.upper(), info["fall_params"], det.threshold, det.window_size, FALL_SAMPLE_FPS))
    return det, flt, info


def main():
    global BENCH
    BENCH = Bench()                                 # ghi RAM hệ thống TRƯỚC khi nạp model
    estimator, cam, model_path = open_model()
    CURRENT_MODEL[0] = model_path
    MODEL_INFO.update(model_info(model_path, estimator))
    det, flt, fall_info = open_fall_detector()
    MODEL_INFO.update(fall_info)
    if CPU_BENCH_ON_START and det is not None:
        try:
            from src import cpu_bench
            from src.shiftgcn_np import ShiftGCN
            d = os.path.join(PROJECT_DIR, "model_classify")
            npz = sorted(f for f in os.listdir(d) if f.endswith(".npz") and not f.endswith("_check.npz"))[0]
            cpu_net = ShiftGCN.load(os.path.join(d, npz))
            chk = os.path.join(d, "shiftgcn_npu_check.npz")
            import numpy as _np
            clips = _np.load(chk)["clip"] if os.path.isfile(chk) else [_np.zeros((17, 17, 3), _np.float32)]
            print("Đo tốc độ CPU cho Shift-GCN (vài giây)...")
            cb = cpu_bench.run(cpu_net, clips)
            MODEL_INFO.update({"cpu_" + k: v for k, v in cb.items()})
            print("[cpu_bench] %s" % json.dumps(cb))
        except Exception as e:
            print("[cpu_bench] lỗi: %s: %s" % (type(e).__name__, e))
    fall_last = None
    next_sample = 0.0
    disp = display.Display() if SHOW_ON_SCREEN else None

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ip = local_ip()
    print("Stream: http://%s:%d/   |   web PC: maix://%s" % (ip, PORT, ip))
    print("Mọi địa chỉ của MaixCAM: %s" % (", ".join(all_ips()) or "chưa có mạng"))
    print("Model đang dùng: %s" % model_path)

    idx, fps_n, fps_t0 = 0, 0, time.time()
    log_t = time.time()
    t_loop = time.perf_counter()
    try:
        while not app.need_exit():
            t_read = time.perf_counter()
            img = cam.read()
            read_ms = (time.perf_counter() - t_read) * 1000.0
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
                CURRENT_MODEL[0] = model_path
                MODEL_INFO.clear()
                MODEL_INFO.update(model_info(model_path, estimator))
                BENCH.reset()
                print("Model đang dùng: %s" % model_path)
                continue
            pose_ms = (time.perf_counter() - t0) * 1000.0
            LATEST.pose_ms = pose_ms

            # ---- phát hiện ngã trên board, lấy mẫu FALL_SAMPLE_FPS (giống main.py nhánh final)
            # Lịch cố định: mốc kế tiếp = mốc trước + 1/FPS (KHÔNG đặt lại = t, nếu không khoảng cách luôn > 100 ms
            # vì phải chờ vòng lặp kế tiếp -> chỉ ~7.7 lần/s khi vòng ~49 ms). Trễ quá 1 chu kỳ thì bắt nhịp lại.
            if det is not None and t >= next_sample:
                period = 1.0 / FALL_SAMPLE_FPS
                next_sample = next_sample + period if t - next_sample < period else t + period
                t_f = time.perf_counter()
                fpose = flt.smooth(pose) if flt is not None else pose
                filter_ms = (time.perf_counter() - t_f) * 1000.0
                res = det.update(fpose)
                fall_ms = (time.perf_counter() - t_f) * 1000.0
                ran = res is not None and res["model_ms"] > 0
                BENCH.add_fall(fall_ms=fall_ms, filter_ms=filter_ms,
                               model_ms=res["model_ms"] if ran else None,
                               prep_ms=res["prep_ms"] if ran else None,
                               post_ms=res["post_ms"] if res is not None else None,
                               e2e_ms=read_ms + pose_ms + fall_ms)
                if res is not None and res["label"] and not (fall_last and fall_last["label"]):
                    BENCH.alerts += 1
                    print("[NGÃ] %s  P(ngã) = %.2f" % (time.strftime("%H:%M:%S"), res["prob"]))
                fall_last = res

            t_enc = time.perf_counter()
            if LATEST.clients > 0:                  # không ai xem thì không tốn công nén JPEG
                jpeg = to_jpeg_bytes(img)          # ảnh SẠCH (chưa vẽ): PC tự vẽ khung xương
                if jpeg:
                    LATEST.publish(jpeg, pose_to_json(pose, idx, t, img.width(), img.height(), pose_ms,
                                                      fall_last))
            encode_ms = (time.perf_counter() - t_enc) * 1000.0

            fps_n += 1
            if t - fps_t0 >= 1.0:
                LATEST.fps = fps_n / (t - fps_t0)
                fps_n, fps_t0 = 0, t
                if ip == "?":
                    ip = local_ip()

            t_disp = time.perf_counter()
            if disp is not None:
                if SHOW_SKELETON:
                    draw_pose(img, pose)
                    img.draw_string(4, 4, "FPS %.1f  %s" % (LATEST.fps, "person" if pose.valid else "no person"),
                                    color=image.COLOR_WHITE)
                    img.draw_string(4, 22, "maix://%s  (%d client)" % (ip, LATEST.clients), color=image.COLOR_GREEN)
                y = 40 if SHOW_SKELETON else 4
                if fall_last is not None:
                    if fall_last["label"]:
                        img.draw_string(4, y, "NGA! P=%.2f" % fall_last["prob"], color=image.COLOR_RED, scale=1.6)
                    else:                             # kể cả khi bị cổng tốc độ chặn (nằm xuống chậm) = không ngã
                        img.draw_string(4, y, "KHONG NGA P=%.2f" % fall_last["prob"], color=image.COLOR_GREEN,
                                        scale=1.6)
                disp.show(img)
            now = time.perf_counter()
            BENCH.add(pose.valid, read_ms=read_ms, pose_ms=pose_ms, encode_ms=encode_ms,
                      display_ms=(now - t_disp) * 1000.0, loop_ms=(now - t_loop) * 1000.0)
            t_loop = now
            if t - log_t >= 10.0:                   # in số liệu đo ra log mỗi 10 s
                log_t = t
                s = BENCH.summary()
                print("[bench] %d khung | pose %.1f ms (p95 %.1f) | ngã %.1f ms (model %.1f) | "
                      "camera->quyết định %.1f ms | vòng %.1f ms -> %.1f FPS | "
                      "RAM tiến trình peak %s MB | RAM hệ thống peak %s / %s MB"
                      % (s["frames"], s.get("pose_ms_mean", 0), s.get("pose_ms_p95", 0),
                         s.get("fall_ms_mean", 0), s.get("model_ms_mean", 0), s.get("e2e_ms_mean", 0),
                         s.get("loop_ms_mean", 0), s.get("fps") or 0, s.get("vm_hwm_mb"),
                         s.get("mem_used_peak_mb"), s.get("mem_total_mb")))
                hz = s.get("sample_hz")
                if hz is not None and hz < 0.9 * FALL_SAMPLE_FPS:
                    print("[CẢNH BÁO] Bộ phân loại chỉ nhận %.1f khung/s (cần %g): cửa sổ dài %.1f s thay vì %.1f s "
                          "-> kết quả ngã KHÔNG tin cậy. Tắt SHOW_ON_SCREEN / đóng web đang xem để tăng FPS."
                          % (hz, FALL_SAMPLE_FPS, 17 / hz, 17 / FALL_SAMPLE_FPS))
    finally:
        server.shutdown()
        estimator.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
