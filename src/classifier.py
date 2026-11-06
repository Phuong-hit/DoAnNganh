"""
classifier.py
-------------
Bộ phân loại ngã DÙNG CHUNG cho mọi kiến trúc (ST-GCN, LSTM, ...).

Hợp đồng chung cho mọi model (để web, dataset, suy luận không phải sửa khi đổi model):
    đầu vào  x [N, C, T, V]  (C=3: x, y đã chuẩn hóa + mask; V=17)  — tạo bằng to_model_input()
    đầu ra   logits [N, 2]   (0 = KHÔNG NGÃ, 1 = NGÃ)
Model nào cần dạng khác (LSTM cần [N, T, V*C]) thì tự đổi trong forward().

Checkpoint (.pt) là dict:
    {"arch": "stgcn" | "lstm" | ..., "state": state_dict,
     "in_channels": 3, "num_classes": 2, "model_kwargs": {...},
     "window_size": 20, "fps": 10.0, ...}
Checkpoint cũ của train.py (không có "arch") được coi là "stgcn".
Khóa tùy chọn "threshold": ngưỡng chọn trên tập val; FallClassifier dùng khi không truyền ngưỡng.

Thêm kiến trúc mới (ví dụ LSTM ở B1):
    # src/model_lstm.py
    from .classifier import register_model
    @register_model("lstm")
    class LSTMClassifier(nn.Module): ...
Tệp src/model_lstm.py đã có sẵn tên trong _BUILTIN_MODULES nên tự được nạp khi tồn tại.
Lưu bằng save_checkpoint(..., arch="lstm").
"""
from __future__ import annotations

import importlib
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .features import normalize_clip
from .motion import SpeedGate, motion_features
from .skeleton_io import from_poses

LABEL_NAMES = {0: "KHÔNG NGÃ", 1: "NGÃ"}

MODEL_REGISTRY: dict[str, type] = {}

# Module chứa model, được import khi cần để tự đăng ký. Thiếu file thì bỏ qua (model_lstm: B1).
_BUILTIN_MODULES = ("model_stgcn", "model_lstm", "model_shiftgcn")


def register_model(name: str):
    """Decorator: @register_model("lstm") class LSTMClassifier(nn.Module)."""
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return deco


def _load_builtin_models():
    for mod in _BUILTIN_MODULES:
        try:
            importlib.import_module(f"{__package__}.{mod}")
        except ModuleNotFoundError as e:
            if e.name != f"{__package__}.{mod}":    # file có nhưng import bên trong lỗi -> báo thật
                raise


def build_model(arch: str, in_channels: int = 3, num_classes: int = 2, **kwargs) -> torch.nn.Module:
    _load_builtin_models()
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Kiến trúc '{arch}' chưa đăng ký. Có: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[arch](in_channels=in_channels, num_classes=num_classes, **kwargs)


def to_model_input(X) -> torch.Tensor:
    """numpy [N,T,V,C] -> tensor float32 [N,C,T,V] (dạng đầu vào chung của mọi model)."""
    x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    return x.permute(0, 3, 1, 2).contiguous()


def save_checkpoint(path, model: torch.nn.Module, arch: str, in_channels: int = 3,
                    num_classes: int = 2, model_kwargs: dict | None = None,
                    window_size: int | None = None, fps: float | None = None, **meta) -> Path:
    """Lưu checkpoint đúng định dạng FallClassifier đọc được (dùng ở script huấn luyện)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}   # nạp được cả máy không GPU
    ckpt = dict(arch=arch, state=state, in_channels=in_channels,
                num_classes=num_classes, model_kwargs=dict(model_kwargs or {}),
                window_size=window_size, fps=fps, **meta)
    torch.save(ckpt, path)
    return path


def load_checkpoint(path, device="cpu") -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy trọng số: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:                               # torch cũ không có weights_only
        return torch.load(path, map_location=device)


def normalize_checkpoint(ckpt) -> dict:
    """Đưa các định dạng checkpoint khác về định dạng chung {"arch", "state", "model_kwargs", ...}.

    - state_dict trần                         -> arch "stgcn" (checkpoint cũ của code_1)
    - định dạng train Shift-GCN của nhóm      -> arch "shiftgcn"
        {"model_state_dict", "num_frames", "num_point", "in_channels", "num_class", "epoch", "best_val_f1", ...}
        (models/shiftgcn_best.pt; cửa sổ = num_frames khung @ 10 fps, ngưỡng 0.5)
    """
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint không đúng định dạng (cần dict).")
    if "model_state_dict" in ckpt and "num_frames" in ckpt:
        kw = dict(num_point=int(ckpt.get("num_point", 17)), num_frames=int(ckpt["num_frames"]))
        meta = {k: v for k, v in ckpt.items()
                if k not in ("model_state_dict", "optimizer_state_dict", "num_point", "num_frames",
                             "in_channels", "num_class")}
        return dict(meta, arch=str(ckpt.get("arch") or "shiftgcn"), state=ckpt["model_state_dict"],
                    in_channels=int(ckpt.get("in_channels", 3)), num_classes=int(ckpt.get("num_class", 2)),
                    model_kwargs=kw, window_size=kw["num_frames"], fps=ckpt.get("fps", 10.0))
    if "state" not in ckpt and "arch" not in ckpt:   # file chỉ chứa state_dict trần
        return {"state": ckpt}
    return ckpt


class FallClassifier:
    """Suy luận ngã theo cửa sổ trượt trên chuỗi PoseResult.

    update(pose) mỗi khung; cứ stride_frames khung (khi đã đủ window_size khung) thì trả dict kết quả,
    ngược lại trả None. predict_clip() dùng cho cửa sổ có sẵn (đánh giá offline).
    """

    def __init__(self, checkpoint_path, window_size=None, stride_frames=5, device=None,
                 fall_threshold=None, verbose=True, speed_gate: dict | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = normalize_checkpoint(load_checkpoint(self.checkpoint_path, self.device))

        self.arch = str(ckpt.get("arch") or "stgcn")
        self.model = build_model(self.arch, in_channels=int(ckpt.get("in_channels", 3)),
                                 num_classes=int(ckpt.get("num_classes", 2)),
                                 **(ckpt.get("model_kwargs") or {})).to(self.device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

        self.window_size = int(window_size or ckpt.get("window_size") or 20)
        self.stride_frames = int(stride_frames)
        # Ngưỡng: tham số > ngưỡng chọn trên val lúc huấn luyện (lưu trong checkpoint) > 0.5
        self.fall_threshold = float(fall_threshold if fall_threshold is not None
                                    else (ckpt.get("threshold") or 0.5))
        self.meta = {k: v for k, v in ckpt.items() if k != "state"}
        self.train_fps = ckpt.get("fps")
        # Cổng tốc độ (src/motion.py): tham số > giá trị chọn bằng calibrate_gate.py lưu trong checkpoint > tắt
        self.gate = SpeedGate.from_dict(speed_gate if speed_gate is not None else ckpt.get("speed_gate"))
        if self.window_size < 1 or self.stride_frames < 1:
            raise ValueError("window_size và stride_frames phải >= 1.")

        self.buffer: deque = deque(maxlen=self.window_size)
        self.frames_since_prediction = 0
        self.last: dict | None = None
        if verbose:
            print(f"Đã nạp {self.arch}: {self.checkpoint_path} (cửa sổ {self.window_size} khung, "
                  f"thiết bị {self.device})")

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def name(self) -> str:
        return f"{self.arch}:{self.checkpoint_path.stem}"

    def reset(self):
        """Xóa lịch sử khi chuyển sang video hoặc người mới."""
        self.buffer.clear()
        self.frames_since_prediction = 0
        self.last = None

    def _result(self, p_fall: float, speed: dict | None = None) -> dict:
        model_label = int(p_fall >= self.fall_threshold)
        gate_ok = True if speed is None else bool(self.gate.passes(speed, p_fall))
        label = int(model_label and gate_ok)
        return {"label": label, "name": LABEL_NAMES[label], "probability": p_fall,
                "probabilities": {"khong_nga": 1.0 - p_fall, "nga": p_fall},
                "model_label": model_label,                 # model nói gì (chưa qua cổng tốc độ)
                "gated": bool(model_label and not gate_ok), # model báo ngã nhưng chuyển động chậm -> bỏ
                "speed": {k: round(float(v), 3) for k, v in (speed or {}).items()}}

    def gate_labels(self, clips, p_fall, fps: float | None = None):
        """Offline: clips [N,T,17,3] đã chuẩn hóa + P(ngã) [N] -> (nhãn cuối [N], đặc trưng tốc độ dict)."""
        feats = motion_features(clips, fps or self.train_fps or 10.0)
        ok = self.gate.passes(feats, p_fall)
        return ((np.asarray(p_fall) >= self.fall_threshold) & ok).astype(np.int64), feats

    @torch.inference_mode()
    def predict_batch(self, clips) -> np.ndarray:
        """clips [N,T,17,3] ĐÃ chuẩn hóa (normalize_clip) -> p_fall [N]."""
        if len(clips) == 0:
            return np.zeros(0, np.float32)
        logits = self.model(to_model_input(clips).to(self.device))
        return torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()

    def predict_clip(self, seq, mask, aspect=1.0, fps: float | None = None) -> dict:
        clip = normalize_clip(seq, mask, aspect=aspect)
        p = float(self.predict_batch(clip[np.newaxis])[0])
        return self._result(p, motion_features(clip, fps or self.train_fps or 10.0))

    def update(self, pose):
        self.buffer.append(pose)
        self.frames_since_prediction += 1
        if len(self.buffer) < self.window_size or self.frames_since_prediction < self.stride_frames:
            return None
        self.frames_since_prediction = 0
        item = from_poses(list(self.buffer))       # cùng quy ước với lúc tạo dataset
        self.last = self.predict_clip(item["seq"], item["mask"], item.get("aspect", 1.0), fps=item.get("fps"))
        return self.last

    def flush(self):
        """Gọi khi video kết thúc. Video ngắn hơn cửa sổ (clip ~1.9 s = 19 khung < 20) thì update() chưa dự đoán
        lần nào: dự đoán một lần trên phần đã gom, đệm cuối cho đủ cửa sổ (giống make_windows lúc tạo dataset).
        Trả kết quả mới, hoặc None nếu đã có dự đoán / chưa có khung nào."""
        if self.last is not None or not self.buffer:
            return None
        item = from_poses(list(self.buffer))
        seq, mask = pad_to_window(item["seq"], item["mask"], self.window_size)
        self.last = self.predict_clip(seq, mask, item.get("aspect", 1.0), fps=item.get("fps"))
        return self.last


def pad_to_window(seq, mask, window_size):
    """Đệm cuối bằng 0 (mask=False) cho đủ window_size khung — đúng cách skeleton_io.make_windows đệm video ngắn."""
    pad = window_size - len(seq)
    if pad <= 0:
        return seq, mask
    seq = np.pad(seq, [(0, pad)] + [(0, 0)] * (np.ndim(seq) - 1))
    mask = np.pad(mask, [(0, pad)] + [(0, 0)] * (np.ndim(mask) - 1))
    return seq, mask
