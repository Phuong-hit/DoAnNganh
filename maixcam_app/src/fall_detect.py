"""Phát hiện ngã TRÊN MaixCAM: cửa sổ skeleton -> chuẩn hóa -> Shift-GCN (numpy) -> hậu xử lý nhánh final.

Bản board của (trên PC):
  src/skeleton_io.py  from_poses / _build_mask      -> build_window()
  src/features.py     normalize_clip                -> normalize_clip()  (vector hóa bằng numpy)
  src/fall_postprocess.py FallPostProcessor         -> FallPostProcessor
  src/classifier.py   FallClassifier._update_post   -> FallDetector.update()
Phép tính giữ nguyên để kết quả board == web (kiểm tra bằng tests/test_maixcam_fall.py trên PC).

Dùng:
    det = FallDetector.load("model_classify/shiftgcn_best_4_20_11.npz")
    res = det.update(pose)      # gọi với khung ĐÃ lấy mẫu 10 FPS + đã qua SkeletonFilter
    # None khi chưa đủ cửa sổ; dict(prob, label, held, model_ms, step_ms) khi có kết quả
"""
import time
from collections import deque

import numpy as np

from .constants import BONES, HIPS
from .shiftgcn_np import ShiftGCN

_BONE_A = np.array([a for a, _ in BONES], dtype=np.int64)
_BONE_B = np.array([b for _, b in BONES], dtype=np.int64)


def build_window(poses):
    """list[PoseResult] -> (seq [T,17,3], mask [T,17], valid [T], aspect). Giống from_poses + _build_mask."""
    raw = np.stack([np.asarray(p.joints, dtype=np.float32) for p in poses])
    conf = np.nan_to_num(raw[:, :, 2], nan=0.0)
    mask = np.isfinite(raw[:, :, 0]) & np.isfinite(raw[:, :, 1]) & (conf > 0)
    seq = np.nan_to_num(raw, nan=0.0)
    seq[:, :, 2] = conf
    seq[~mask] = 0.0
    valid = np.array([bool(p.valid) for p in poses])
    aspect = 1.0
    for p in poses:
        if p.width and p.height:
            aspect = float(p.width) / float(p.height)
            break
    return seq, mask, valid, aspect


def pose_row(pose):
    """Một PoseResult -> (seq [17,3], mask [17], valid, aspect): giống build_window([pose]) nhưng tính 1 lần / khung."""
    raw = np.asarray(pose.joints, dtype=np.float32)
    conf = np.nan_to_num(raw[:, 2], nan=0.0)
    mask = np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1]) & (conf > 0)
    seq = np.nan_to_num(raw, nan=0.0)
    seq[:, 2] = conf
    seq[~mask] = 0.0
    aspect = float(pose.width) / float(pose.height) if pose.width and pose.height else None
    return seq, mask, bool(pose.valid), aspect


def normalize_clip(seq, mask, aspect=1.0):
    """[W,17,>=2] + mask [W,17] -> [W,17,3]: x, y (một tâm + một tỉ lệ cho cả clip), kênh 3 = mask."""
    seq = np.asarray(seq, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(seq.shape[:2] + (3,), dtype=np.float32)
    if not mask.any():
        return out
    xy = seq[..., :2].copy()
    xy[..., 0] *= aspect
    hip_ok = mask[:, HIPS]
    if hip_ok.any():
        center = xy[:, HIPS][hip_ok].mean(axis=0)
    else:
        center = xy[mask].mean(axis=0)
    both = mask[:, _BONE_A] & mask[:, _BONE_B]                       # [W, số xương]
    if both.any():
        d = xy[:, _BONE_A] - xy[:, _BONE_B]
        lengths = np.sqrt((d * d).sum(axis=-1))[both]
        scale = float(np.median(lengths))
    else:
        scale = 0.0
    if scale < 1e-3:
        scale = 1.0
    xy = (xy - center) / scale
    xy[~mask] = 0.0
    out[..., :2] = xy
    out[..., 2] = mask
    return out


def is_standing_upright(seq, mask, aspect=1.0):
    """seq [17,>=2], mask [17] của MỘT khung -> cả người (thân + chân) đã đứng thẳng chưa (giống nhánh final)."""
    sh = [i for i in (5, 6) if mask[i]]
    hp = [i for i in (11, 12) if mask[i]]
    kn = [i for i in (13, 14) if mask[i]]
    an = [i for i in (15, 16) if mask[i]]
    if not sh or not hp or (not an and not kn):
        return False

    def mx(ids):
        return sum(float(seq[i][0]) for i in ids) / len(ids) * aspect

    def my(ids):
        return sum(float(seq[i][1]) for i in ids) / len(ids)

    torso_dx = abs(mx(sh) - mx(hp))
    torso_dy = my(hp) - my(sh)
    if torso_dy <= 0 or torso_dy <= torso_dx * 2.0:
        return False
    low = an if an else kn
    leg_dx = abs(mx(hp) - mx(low))
    leg_dy = my(low) - my(hp)
    if leg_dy <= 0 or leg_dy <= leg_dx * 2.0:
        return False
    return leg_dy >= torso_dy * (0.85 if an else 0.60)


def speed_ok(clip, gate, fps=10.0, span=3):
    """Cổng tốc độ (giống src/motion.py motion_features + SpeedGate, không có keep_if_p_above).
    clip [T,17,3] đã chuẩn hóa -> True khi vai hạ >= gate["shoulder_drop"] thân/s HOẶC thân xoay >= gate["rotation"]
    độ/s (lấy lớn nhất trên các đoạn span khung). gate None -> True (tắt)."""
    if not gate:
        return True
    m = clip[..., 2] > 0
    ok_sh = m[:, [5, 6]].all(1)
    ok_hp = m[:, [11, 12]].all(1)
    sh = np.where(ok_sh[:, None], clip[:, [5, 6], :2].mean(1), np.nan)
    hp = np.where(ok_hp[:, None], clip[:, [11, 12], :2].mean(1), np.nan)
    torso = sh - hp
    angle = np.degrees(np.arctan2(np.abs(torso[:, 0]), -torso[:, 1]))

    def peak(x):
        if len(x) <= span:
            return 0.0
        r = (x[span:] - x[:-span]) / span * fps
        r = r[np.isfinite(r)]
        return float(r.max()) if len(r) else 0.0

    return (peak(sh[:, 1]) >= float(gate.get("shoulder_drop", 2.0))
            or peak(angle) >= float(gate.get("rotation") or 1e9))


class NPUShiftGCN:
    """Shift-GCN chạy trên NPU (shiftgcn_npu.cvimodel, tạo bởi maixcam_app/convert_shiftgcn.sh).
    Cùng giao diện với ShiftGCN (numpy): predict(clip [T,V,3]) -> P(ngã); meta / T / num_params lấy từ bản numpy."""

    def __init__(self, mud_path, ref):
        from maix import nn, tensor
        self._tensor = tensor
        self.model = nn.NN(mud_path, dual_buff=False)
        self.in_name = self.model.inputs_info()[0].name
        self.out_name = self.model.outputs_info()[-1].name
        self.meta, self.T, self.threshold = ref.meta, ref.T, ref.threshold
        self.num_params = ref.num_params
        self.backend = "npu"

    def logits(self, clip):
        x = np.ascontiguousarray(np.asarray(clip, dtype=np.float32).transpose(2, 0, 1)[None])   # [1, 3, T, V]
        ins = self._tensor.Tensors()
        ins.add_tensor(self.in_name, self._tensor.tensor_from_numpy_float32(x), True, True)
        outs = self.model.forward(ins, copy_result=True)
        return np.asarray(self._tensor.tensor_to_numpy_float32(outs[self.out_name]), dtype=np.float64).reshape(-1)

    def predict(self, clip):
        z = self.logits(clip)
        z = z - z.max()
        e = np.exp(z)
        return float(e[1] / e.sum())


def load_net(npz_path, npu_mud=None, check_npz=None, max_err=0.05):
    """Ưu tiên NPU; lỗi nạp / tự kiểm tra không đạt -> bản numpy (CPU). Trả (net, info dict cho log và /bench)."""
    ref = ShiftGCN.load(npz_path)
    ref.backend = "numpy"
    info = {"fall_backend": "numpy"}
    if not npu_mud:
        return ref, info
    try:
        npu = NPUShiftGCN(npu_mud, ref)
        if check_npz:                                  # 8 clip thật + P(ngã) PyTorch, tính sẵn trên PC
            d = np.load(check_npz, allow_pickle=False)
            p_ref = d["p"]
            p_npu = np.array([npu.predict(c) for c in d["clip"]])
            err = float(np.abs(p_npu - p_ref).max())
            flips = int(((p_npu >= ref.threshold) != (p_ref >= ref.threshold)).sum())
            info.update(fall_npu_check_err=round(err, 4), fall_npu_check_flips=flips)
            if err > max_err or flips:
                info["fall_npu_error"] = "tự kiểm tra lệch %.3f, đổi %d nhãn" % (err, flips)
                return ref, info
        t0 = time.perf_counter()
        npu.predict(np.zeros((ref.T, 17, 3), np.float32))
        info.update(fall_backend="npu", fall_npu_first_ms=round((time.perf_counter() - t0) * 1000.0, 2))
        return npu, info
    except Exception as e:                              # thiếu API / model lỗi -> vẫn chạy bằng numpy
        info["fall_npu_error"] = "%s: %s" % (type(e).__name__, e)
        return ref, info


class FallPostProcessor:
    """Giống src/fall_postprocess.py (kể cả max_hold_frames: nhả cảnh báo khi P thấp liên tục N khung)."""

    def __init__(self, threshold=0.7, consec_frames=1, hold_frames=10, max_hold_frames=None):
        self.threshold = threshold
        self.consec_frames = consec_frames
        self.hold_frames = hold_frames
        self.max_hold_frames = max_hold_frames
        self.streak = 0
        self.hold_counter = 0
        self.low_in_hold = 0

    def update(self, raw_prob, seq=None, mask=None, aspect=1.0, fast=True):
        self.streak = self.streak + 1 if (raw_prob >= self.threshold and fast) else 0
        if self.streak >= self.consec_frames:
            self.hold_counter = self.hold_frames
        if self.hold_counter > 0:
            self.low_in_hold = self.low_in_hold + 1 if raw_prob < self.threshold else 0
            if self.max_hold_frames and self.low_in_hold >= self.max_hold_frames:
                self.hold_counter = 0
                self.low_in_hold = 0
                return False
            if seq is not None and mask is not None and is_standing_upright(seq, mask, aspect):
                self.hold_counter = max(0, self.hold_counter - 3)
            else:
                self.hold_counter = max(self.hold_counter, 3)
            return True
        self.low_in_hold = 0
        return False

    def reset(self):
        self.streak = 0
        self.hold_counter = 0
        self.low_in_hold = 0


class FallDetector:
    """Vòng lặp giống FallClassifier._update_post (web) / main.py nhánh final."""

    def __init__(self, net, threshold=None, window_size=None, stride_frames=None, postprocess=None):
        m = net.meta
        self.net = net
        self.threshold = float(threshold if threshold is not None else m.get("threshold", 0.5))
        self.window_size = int(window_size or m.get("window_size") or net.T)
        self.stride_frames = int(stride_frames or m.get("stride_frames") or 1)
        pp = postprocess if postprocess is not None else (m.get("postprocess") or {})
        self.min_valid_ratio = float(pp.get("min_valid_ratio", 0.5))
        self.reset_empty_frames = int(pp.get("reset_empty_frames", 5))
        mh = pp.get("max_hold_frames")
        self.gate = pp.get("speed_gate")                       # None = tắt; dict(shoulder_drop, rotation)
        self.fps = float(m.get("fps") or 10.0)
        self.post = FallPostProcessor(self.threshold, int(pp.get("consec_frames", 1)), int(pp.get("hold_frames", 10)),
                                      int(mh) if mh else None)
        self.buffer = deque(maxlen=self.window_size)
        self.reset()

    @classmethod
    def load(cls, path, npu_mud=None, check_npz=None, **kw):
        """path = .npz (trọng số numpy, luôn cần: meta + dự phòng); npu_mud = shiftgcn_npu.mud nếu có."""
        net, info = load_net(path, npu_mud, check_npz)
        det = cls(net, **kw)
        det.load_info = info
        return det

    @property
    def name(self):
        return "shiftgcn:" + str(self.net.meta.get("source", "")).rsplit(".", 1)[0]

    def reset(self):
        self.buffer.clear()
        self.post.reset()
        self.raw = 0.0
        self.fast = True
        self.empty = 0
        self.full_count = 0
        self.aspect = 1.0
        self.last = None

    def update(self, pose):
        t0 = time.perf_counter()
        model_ms = 0.0
        if not pose.valid:
            self.empty += 1
            if self.empty >= self.reset_empty_frames:          # mất người lâu: xóa cửa sổ + cảnh báo
                self.buffer.clear()
                self.post.reset()
                self.full_count = 0
                self.last = None
        else:
            self.empty = 0
        row = pose_row(pose)                                   # đổi numpy 1 lần / khung, giữ lại trong cửa sổ
        self.buffer.append(row)
        if len(self.buffer) < self.window_size:
            return None
        t_w = time.perf_counter()
        prep_ms = 0.0
        if self.full_count % self.stride_frames == 0:
            rows = list(self.buffer)
            self.aspect = next((r[3] for r in rows if r[3]), 1.0)          # giống from_poses: khung đầu có kích thước
            n_valid = sum(1 for r in rows if r[2])
            if n_valid / float(len(rows)) >= self.min_valid_ratio:
                clip = normalize_clip(np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]), self.aspect)
                t1 = time.perf_counter()
                prep_ms = (t1 - t_w) * 1000.0
                self.raw = self.net.predict(clip)
                model_ms = (time.perf_counter() - t1) * 1000.0
                self.fast = speed_ok(clip, self.gate, self.fps)
        self.full_count += 1
        t_p = time.perf_counter()
        if pose.valid:
            label = self.post.update(self.raw, row[0], row[1], self.aspect, self.fast)
        else:
            label = self.post.update(self.raw, None, None, self.aspect, self.fast)
        now = time.perf_counter()
        self.last = dict(prob=self.raw, label=int(label), held=bool(label and self.raw < self.threshold),
                         gated=bool(not label and self.raw >= self.threshold and not self.fast),
                         model_ms=model_ms, prep_ms=prep_ms, post_ms=(now - t_p) * 1000.0,
                         step_ms=(now - t0) * 1000.0)
        return self.last
