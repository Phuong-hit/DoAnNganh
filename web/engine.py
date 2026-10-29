"""Ghép các khối của src/ thành pipeline cho web:

    FileSource / StreamSource  ->  PoseEstimate  ->  SkeletonFilter  ->  CSVExporter (+ NPZ) / SkeletonDraw
                                                                    ->  FallClassifier (ST-GCN / LSTM, tùy chọn)

Toàn bộ logic xử lý nằm trong src/. File này chỉ:
  - tạo các đối tượng từ cấu hình,
  - chọn FileSource hay StreamSource theo chuỗi nguồn người dùng nhập,
  - đo latency pose / FPS đầu-cuối (src/ không đo),
  - ghi CSV + NPZ vào file tạm rồi mới đổi tên, để video trích dở không bị coi là "đã có".
    NPZ (skeleton_io.from_poses) giữ aspect của khung -> là đầu vào của bước tạo dataset/huấn luyện.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import cv2
import numpy as np

from src.csv_export import CSVExporter
from src.pose_estimate import PoseEstimate, PoseResult
from src.skeleton_io import from_poses
from src.skeleton_draw import SkeletonDraw
from src.skeleton_filter import SkeletonFilter
from src.video_process import FileSource
from src.webcam import StreamSource

from .config import resolve

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def make_video_id(text: str) -> str:
    """Tên file -> video_id an toàn (chỉ chữ, số, _). Cũng là tên file CSV."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_") or "video"


# ---------------------------------------------------------------------------- tạo đối tượng từ cấu hình
def build_estimator(cfg: dict) -> PoseEstimate:
    p = cfg["pose"]
    kw = dict(kp_conf=p.get("kp_conf", 0.35), min_valid_joints=p.get("min_valid_joints", 5),
              track_iou=p.get("track_iou", 0.1), lost_reset_frames=p.get("lost_reset_frames", 30))
    backend = p.get("backend", "yolo")
    if backend == "dummy":
        from .dummy_backend import DummyBackend
        return PoseEstimate(backend=DummyBackend(), **kw)
    if backend != "yolo":
        raise ValueError(f"pose.backend không hợp lệ: {backend} (yolo | dummy)")
    model = Path(str(p.get("model_path", "yolov8n-pose.pt")))
    # Tên trần (yolov8n-pose.pt) để ultralytics tự tải; có thư mục thì tính từ gốc project.
    if model.parent != Path("."):
        model = resolve(model)
    return PoseEstimate(model_path=model, conf=p.get("conf", 0.4), device=p.get("device"),
                        iou=p.get("iou", 0.5), use_tracker=p.get("use_tracker", True),
                        tracker=p.get("tracker", "bytetrack.yaml"), imgsz=p.get("imgsz"), **kw)


def build_filter(cfg: dict, enabled: bool | None = None) -> SkeletonFilter | None:
    f = dict(cfg.get("filter") or {})
    on = f.pop("enabled", True) if enabled is None else enabled
    f.pop("enabled", None)
    return SkeletonFilter(**f) if on else None


def build_drawer(cfg: dict) -> SkeletonDraw:
    d = cfg.get("draw") or {}
    return SkeletonDraw(min_conf=d.get("min_conf", 0.35), show_bbox=d.get("show_bbox", True),
                        show_score=d.get("show_score", True), show_kp_conf=d.get("show_kp_conf", False))


# Thứ tự ưu tiên khi classifier.checkpoint = auto: LSTM (B1) trước, rồi tới ST-GCN của code_1.
AUTO_CHECKPOINTS = ("models/best_lstm.pt", "models/best_stgcn.pt")


def list_checkpoints(cfg: dict | None = None) -> list[str]:
    """Các checkpoint phân loại ngã trong models/ (bỏ qua trọng số YOLO)."""
    d = resolve("models")
    if not d.is_dir():
        return []
    return sorted(f"models/{p.name}" for p in d.glob("*.pt")
                  if "yolo" not in p.name.lower() and "pose" not in p.name.lower())


def classifier_path(cfg: dict) -> Path | None:
    """Đường dẫn checkpoint của bộ phân loại ngã, None nếu chưa cấu hình / không tìm thấy.
    "auto": models/best_lstm.pt nếu có, không thì models/best_stgcn.pt."""
    ck = (cfg.get("classifier") or {}).get("checkpoint")
    if not ck:
        return None
    if str(ck).lower() == "auto":
        for c in AUTO_CHECKPOINTS:
            if resolve(c).is_file():
                return resolve(c)
        return None
    return resolve(ck)


def build_classifier(cfg: dict):
    """FallClassifier (ST-GCN / LSTM...) theo cấu hình, None nếu chưa cấu hình checkpoint.
    torch chỉ được import ở đây, nên web vẫn chạy khi chưa cài torch / chưa có model."""
    path = classifier_path(cfg)
    if path is None:
        return None
    from src.classifier import FallClassifier
    c = cfg.get("classifier") or {}
    return FallClassifier(path, window_size=c.get("window_size"), stride_frames=c.get("stride_frames", 5),
                          device=c.get("device"), fall_threshold=c.get("threshold"), verbose=False)


def pose_tag(cfg: dict) -> str:
    p = cfg["pose"]
    if p.get("backend", "yolo") == "dummy":
        return "dummy"
    return Path(str(p.get("model_path", "yolo"))).stem


def config_tag(cfg: dict) -> str:
    """Tên thư mục kết quả theo cấu hình, ví dụ 'yolov8n-pose_10fps_filt'.
    Đổi model / FPS / bật-tắt lọc thì kết quả nằm ở thư mục khác, không bao giờ lẫn."""
    fps = cfg["source"].get("target_fps")
    fps_s = f"{fps:g}fps" if fps else "nativefps"
    filt = "filt" if (cfg.get("filter") or {}).get("enabled", True) else "raw"
    return f"{pose_tag(cfg)}_{fps_s}_{filt}"


def output_dir(cfg: dict) -> Path:
    return resolve(cfg["paths"]["output"]) / config_tag(cfg)


def csv_dir(cfg: dict) -> Path:
    # CSVExporter(video_id, output_dir) ghi vào <output_dir>/csv/<video_id>.csv
    return output_dir(cfg) / "csv"


def npz_dir(cfg: dict) -> Path:
    """<output_dir>/npz/<video_id>.npz: seq, mask, valid, t, frame_idx, fps, aspect (skeleton_io.from_poses)."""
    return output_dir(cfg) / "npz"


def npz_path_for(csv_path: Path) -> Path:
    """.../csv/<id>.csv -> .../npz/<id>.npz (cùng cấu hình)."""
    csv_path = Path(csv_path)
    return csv_path.parent.parent / "npz" / f"{csv_path.stem}.npz"


def save_npz(path: Path, poses: list[PoseResult], video_id: str):
    """Lưu chuỗi pose theo quy ước skeleton_io (ghi file tạm rồi đổi tên)."""
    item = from_poses(poses)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{uuid.uuid4().hex[:8]}.npz")
    try:
        np.savez_compressed(tmp, video_id=np.array(video_id), **item)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    for k in ("fps", "aspect"):
        d[k] = float(d[k])
    d["video_id"] = str(d.get("video_id", Path(path).stem))
    return d


# ---------------------------------------------------------------------------- nguồn
class OpenedSource:
    """Bọc FileSource / StreamSource của src/ để web dùng chung một kiểu.

    FileSource và StreamSource luôn tạo thư mục <output_dir>/frames/<video_id>/ ngay khi khởi tạo
    (web không lưu ảnh), nên ở đây truyền vào một thư mục tạm và xóa khi đóng.
    """

    def __init__(self, spec: str, cfg: dict, video_id: str | None = None):
        self._scratch = tempfile.mkdtemp(prefix="fallweb_")
        fps = cfg["source"].get("target_fps")
        text = str(spec).strip()
        try:
            if text.isdigit() or text.lower().startswith(STREAM_PREFIXES):
                cam = int(text) if text.isdigit() else text
                self.video_id = video_id or (f"webcam{cam}" if isinstance(cam, int)
                                             else "stream_" + make_video_id(urlparse(text).hostname or "url"))
                self.src = StreamSource(cam, output_dir=self._scratch, target_fps=float(fps or 10.0),
                                        video_id=self.video_id)
                self.is_live = True
                self.kind = "stream"
            else:
                path = Path(text)
                if not path.is_absolute() and not path.exists():
                    path = resolve(path)
                if not path.is_file():
                    raise FileNotFoundError(f"Không tìm thấy nguồn: {spec}")
                self.video_id = video_id or make_video_id(path.stem)
                self.src = FileSource(path, output_dir=self._scratch, target_fps=fps)
                self.is_live = False
                self.kind = "file"
        except Exception:
            shutil.rmtree(self._scratch, ignore_errors=True)
            raise

    @property
    def out_fps(self) -> float:
        return float(self.src.out_fps)

    @property
    def expected_frames(self) -> int | None:
        """Số khung sau lấy mẫu dự kiến (chỉ video file), để hiện tiến độ."""
        if self.kind != "file":
            return None
        n, sfps = self.src.total_src_frames, self.src.src_fps
        return int(math.ceil(n * self.out_fps / sfps)) if n > 0 and sfps > 0 else None

    def frames(self):
        return self.src.get_frames()

    def close(self):
        try:
            self.src.release()
        finally:
            shutil.rmtree(self._scratch, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------- thống kê
class Stats:
    """Số khung, tỉ lệ có người, latency pose (ms), FPS đầu-cuối."""

    def __init__(self):
        self.lat: list[float] = []
        self.valid = 0
        self.t_start = time.perf_counter()

    def add(self, pose: PoseResult, latency_ms: float):
        self.lat.append(latency_ms)
        self.valid += int(pose.valid)

    @property
    def frames(self) -> int:
        return len(self.lat)

    def summary(self) -> dict:
        wall = time.perf_counter() - self.t_start
        lat = np.array(self.lat) if self.lat else np.zeros(1)
        return dict(frames=self.frames,
                    valid_ratio=round(self.valid / self.frames, 3) if self.frames else 0.0,
                    pose_ms_mean=round(float(lat.mean()), 2),
                    pose_ms_p95=round(float(np.percentile(lat, 95)), 2),
                    e2e_fps=round(self.frames / wall, 2) if wall > 0 else 0.0,
                    wall_s=round(wall, 2))


def estimate(estimator: PoseEstimate, flt: SkeletonFilter | None, frame, idx: int, t: float):
    """Một bước của pipeline: pose (đo thời gian) -> lọc. Trả (pose_đã_lọc, latency_ms)."""
    t0 = time.perf_counter()
    pose = estimator.process_frame(frame, idx, t)
    latency = (time.perf_counter() - t0) * 1000.0
    if flt is not None:
        pose = flt.smooth(pose)
    return pose, latency


def draw_hud(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Chữ thông tin ở góc DƯỚI trái (góc trên trái là chỗ SkeletonDraw ghi "NO PERSON")."""
    scale = max(0.4, image.shape[0] / 900)
    step = int(24 * scale / 0.5)
    y = image.shape[0] - 10 - step * (len(lines) - 1)
    for line in lines:
        cv2.putText(image, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += step
    return image


def draw_fall(image: np.ndarray, result: dict | None, threshold: float = 0.5) -> np.ndarray:
    """Nhãn kết quả phân loại ở góc PHẢI trên: đỏ = NGÃ, xanh = không ngã, xám = đang gom khung."""
    if result is None:
        text, color = "FALL? ...", (128, 128, 128)
    elif result["label"] == 1:
        text, color = f"FALL {result['probability']:.2f}", (0, 0, 255)
    else:
        text, color = f"no fall {result['probability']:.2f}", (0, 160, 0)
    scale = max(0.6, image.shape[0] / 600)
    th = max(2, int(scale * 2))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x2, y1 = image.shape[1] - 8, 8
    cv2.rectangle(image, (x2 - tw - 12, y1), (x2, y1 + tht + 14), color, -1)
    cv2.putText(image, text, (x2 - tw - 6, y1 + tht + 6), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), th, cv2.LINE_AA)
    return image


def predict_video(clf, item: dict, batch: int = 256) -> dict:
    """Chạy bộ phân loại trên cả video đã trích (NPZ), cửa sổ kết thúc ở mỗi stride_frames khung.
    Cùng cách cắt và chuẩn hóa với FallClassifier.update(), để trình phát vẽ được P(ngã) theo thời gian."""
    from src.features import normalize_clip
    W, S = clf.window_size, clf.stride_frames
    seq, mask, t = item["seq"], item["mask"], np.asarray(item["t"], dtype=float)
    ends = list(range(W - 1, len(seq), S))
    clips = [normalize_clip(seq[e - W + 1:e + 1], mask[e - W + 1:e + 1], item["aspect"]) for e in ends]
    probs = []
    for i in range(0, len(clips), batch):
        probs.append(clf.predict_batch(np.stack(clips[i:i + batch])))
    p = np.concatenate(probs) if probs else np.zeros(0)
    return dict(t=[round(float(t[e]), 3) for e in ends], p=[round(float(v), 4) for v in p],
                threshold=clf.fall_threshold, window_size=W, stride_frames=S, model=clf.name)


# ---------------------------------------------------------------------------- trích skeleton -> CSV
class Cancelled(Exception):
    pass


def extract_to_csv(spec: str, video_id: str, out_csv: Path, cfg: dict,
                   estimator: PoseEstimate | None = None,
                   progress: Callable[[int, int | None], None] | None = None,
                   should_stop: Callable[[], bool] | None = None,
                   max_frames: int | None = None, out_npz: Path | None = None) -> dict:
    """Trích skeleton một video, ghi CSV theo đúng định dạng CSVExporter của src/,
    kèm NPZ (skeleton_io.from_poses, có aspect) ở out_npz (mặc định .../npz/<video_id>.npz).

    Ghi vào thư mục tạm cạnh out_csv rồi os.replace: nếu lỗi/hủy giữa chừng thì không để lại
    CSV dở dang.
    """
    own = estimator is None
    estimator = estimator or build_estimator(cfg)
    flt = build_filter(cfg)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = out_csv.parent / f".tmp_{uuid.uuid4().hex[:8]}"
    try:
        with OpenedSource(spec, cfg, video_id=video_id) as src:
            if src.is_live:
                raise ValueError("Trích skeleton chỉ dùng cho video file (webcam/luồng hãy dùng Xem trực tiếp).")
            estimator.reset()                      # không bám nhầm người của video trước
            if flt is not None:
                flt.reset()
            stats = Stats()
            poses: list[PoseResult] = []
            total = src.expected_frames
            with CSVExporter(video_id, tmp_root) as out:
                for pkt in src.frames():
                    if should_stop is not None and should_stop():
                        raise Cancelled()
                    pose, lat = estimate(estimator, flt, pkt.frame, pkt.idx, pkt.t)
                    out.write(pose)
                    poses.append(pose)
                    stats.add(pose, lat)
                    if progress is not None and stats.frames % 5 == 0:
                        progress(stats.frames, total)
                    if max_frames and stats.frames >= max_frames:
                        break
                missing_ratio = out.missing_ratio
                tmp_csv = out.output_path
            if stats.frames == 0:
                raise ValueError("Không đọc được khung hình nào (file hỏng hoặc codec không hỗ trợ).")
            save_npz(Path(out_npz) if out_npz else npz_path_for(out_csv), poses, video_id)
            os.replace(tmp_csv, out_csv)              # CSV đổi tên sau cùng: có CSV <=> có NPZ
            s = stats.summary()
            s.update(missing_ratio=round(missing_ratio, 3), out_fps=round(src.out_fps, 3),
                     filter=flt is not None)
            if progress is not None:
                progress(stats.frames, stats.frames)
            return s
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        if own:
            estimator.close()
