"""Shift-GCN chạy bằng NUMPY (MaixCAM không có PyTorch). Cùng phép tính với src/model_shiftgcn.py trên PC.

Trọng số: file .npz do export_maixcam_model.py (trên PC) tạo ra: meta (JSON) + "w/<tên trong state_dict>".
Một clip [T, V, 3] (đã normalize_clip) -> P(ngã).

Bố cục tensor trong file này: [C, T, V] (một clip, không có chiều batch).
BatchNorm (chế độ eval) được đổi sẵn thành  y = x * scale + shift  lúc nạp.
"""
import json

import numpy as np


def spatial_shift_index(num_nodes, channels, inverse=False):
    size = num_nodes * channels
    sign = -1 if inverse else 1
    i = np.repeat(np.arange(num_nodes), channels)
    j = np.tile(np.arange(channels), num_nodes)
    return ((i * channels + j + sign * j * channels) % size).astype(np.int64)


def temporal_shift_tables(channels, num_frames, kernel=3, reverse=False):
    offsets = (np.arange(channels) % kernel) - kernel // 2
    if reverse:
        offsets = -offsets
    c = np.arange(channels)[:, None]
    t = np.arange(num_frames)[None, :]
    src = t + offsets[:, None]
    valid = (src >= 0) & (src < num_frames)
    idx = np.where(valid, c * num_frames + src, c * num_frames + t)
    return idx.reshape(-1).astype(np.int64), valid.reshape(-1).astype(np.float32)


class ShiftGCN:
    def __init__(self, weights, meta):
        self.meta = meta
        self.V = int(meta["num_point"])
        self.T = int(meta["num_frames"])
        self.C_in = int(meta["in_channels"])
        self.eps = float(meta.get("bn_eps", 1e-5))
        self.threshold = float(meta.get("threshold", 0.5))
        w = weights
        self.w = w

        def bn(prefix):
            g = w[prefix + ".weight"]
            b = w[prefix + ".bias"]
            mu = w[prefix + ".running_mean"]
            var = w[prefix + ".running_var"]
            scale = (g / np.sqrt(var + self.eps)).astype(np.float32)
            return scale, (b - mu * scale).astype(np.float32)

        V, C = self.V, self.C_in
        s, h = bn("data_bn")                                  # BatchNorm1d(V*C), chỉ số v*C + c
        self.data_scale = s.reshape(V, C).T[:, None, :]       # [C, 1, V]
        self.data_shift = h.reshape(V, C).T[:, None, :]

        self.blocks = []
        prev, T = C, self.T
        k = int(meta.get("kernel", 3))
        for i, (ch, st) in enumerate(zip(meta["channels"], meta["strides"])):
            p = "blocks.%d." % i
            blk = {"in": prev, "out": ch, "stride": int(st)}
            # --- SpatialShiftGCN
            g = p + "gcn."
            blk["shift_in"] = spatial_shift_index(V, prev)
            blk["shift_out"] = spatial_shift_index(V, ch, inverse=True)
            blk["mask"] = (np.tanh(w[g + "Feature_Mask"][0]) + 1.0).astype(np.float32)   # [V, in]
            blk["lin_w"] = w[g + "Linear_weight"]                                         # [in, out]
            blk["lin_b"] = w[g + "Linear_bias"].reshape(-1)                               # [out]
            s, h = bn(g + "bn")                                                           # BN1d(V*out)
            blk["gbn"] = (s.reshape(V, ch), h.reshape(V, ch))
            if (g + "down.0.weight") in w:
                s, h = bn(g + "down.1")
                blk["down"] = (w[g + "down.0.weight"][:, :, 0, 0], w[g + "down.0.bias"], s, h)
            else:
                blk["down"] = None
            # --- ShiftTCN
            tc = p + "tcn."
            blk["tbn1"] = bn(tc + "bn")
            blk["tshift_in"] = temporal_shift_tables(ch, T, k)
            blk["t_w"] = w[tc + "temporal_linear.weight"][:, :, 0, 0]
            blk["t_b"] = w[tc + "temporal_linear.bias"]
            blk["tshift_out"] = temporal_shift_tables(ch, T, k, reverse=True)
            blk["tbn2"] = bn(tc + "bn2")
            # --- residual
            if not meta["residual"][i]:
                blk["res"] = None
            elif (p + "residual.0.weight") in w:
                s, h = bn(p + "residual.1")
                blk["res"] = (w[p + "residual.0.weight"][:, :, 0, 0], w[p + "residual.0.bias"], s, h)
            else:
                blk["res"] = "id"
            self.blocks.append(blk)
            prev = ch
            T = (T - 1) // int(st) + 1
        self.fc_w = w["fc.weight"]
        self.fc_b = w["fc.bias"]

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        weights = {k[2:]: d[k].astype(np.float32) for k in d.files if k.startswith("w/")}
        return cls(weights, meta)

    @property
    def num_params(self):
        return int(sum(v.size for k, v in self.w.items()
                       if not (k.endswith("running_mean") or k.endswith("running_var"))))

    # ------------------------------------------------------------------ các bước
    @staticmethod
    def _conv1x1(x, wb):                       # x [C, T, V]; wb = (W [out, in], b, bn_scale, bn_shift)
        W, b, s, h = wb
        y = np.tensordot(W, x, axes=(1, 0)) + b[:, None, None]
        return y * s[:, None, None] + h[:, None, None]

    @staticmethod
    def _tshift(x, table):                     # x [C, T, V]
        idx, valid = table
        c, t, v = x.shape
        y = x.reshape(c * t, v)[idx] * valid[:, None]
        return y.reshape(c, t, v)

    def _gcn(self, x0, blk):
        c, t, v = x0.shape
        x = x0.transpose(1, 2, 0).reshape(t, v * c)[:, blk["shift_in"]].reshape(t, v, c)
        x = x * blk["mask"]
        x = x @ blk["lin_w"] + blk["lin_b"]                          # [T, V, out]
        o = blk["out"]
        x = x.reshape(t, v * o)[:, blk["shift_out"]].reshape(t, v, o)
        s, h = blk["gbn"]
        x = (x * s + h).transpose(2, 0, 1)                          # [out, T, V]
        x = x + (x0 if blk["down"] is None else self._conv1x1(x0, blk["down"]))
        return np.maximum(x, 0.0)

    def _tcn(self, x, blk):
        s, h = blk["tbn1"]
        x = x * s[:, None, None] + h[:, None, None]
        x = self._tshift(x, blk["tshift_in"])
        x = np.tensordot(blk["t_w"], x, axes=(1, 0)) + blk["t_b"][:, None, None]
        x = np.maximum(x, 0.0)
        x = self._tshift(x, blk["tshift_out"])
        if blk["stride"] > 1:
            x = x[:, ::blk["stride"], :]
        s, h = blk["tbn2"]
        return x * s[:, None, None] + h[:, None, None]

    # ------------------------------------------------------------------ bản nhanh cho CPU
    def _prepare_fast(self):
        """Bố cục [T, V, C] liền mạch suốt mạng: phép nhân là ma trận 2D (T*V, C_in) @ (C_in, C_out), dịch thời gian
        là lát cắt bước k (kênh c có độ dịch c % k - k//2), BN sau conv 1x1 gộp sẵn vào trọng số. Cùng phép tính."""
        V, k = self.V, int(self.meta.get("kernel", 3))
        self._k = k
        self.f_data_scale = self.data_scale[:, 0, :].T.copy()            # [V, C]
        self.f_data_shift = self.data_shift[:, 0, :].T.copy()
        fb = []
        for blk in self.blocks:
            ci, co = blk["in"], blk["out"]
            f = {"in": ci, "out": co, "stride": blk["stride"], "shift_in": blk["shift_in"], "shift_out": blk["shift_out"],
                 "mask": blk["mask"], "lin_w": np.ascontiguousarray(blk["lin_w"]), "lin_b": blk["lin_b"],
                 "gs": blk["gbn"][0].reshape(-1), "gh": blk["gbn"][1].reshape(-1)}

            def fold(wb):                                    # (W [out,in], b, s, h) -> W' [in,out], b' [out]
                W, b, s, h = wb
                return np.ascontiguousarray((W * s[:, None]).T), (b * s + h).astype(np.float32)
            f["down"] = None if blk["down"] is None else fold(blk["down"])
            f["s1"], f["h1"] = blk["tbn1"]
            f["t_w"] = np.ascontiguousarray(blk["t_w"].T)                # [in, out]
            f["t_b"] = blk["t_b"]
            f["s2"], f["h2"] = blk["tbn2"]
            f["res"] = blk["res"] if blk["res"] in (None, "id") else fold(blk["res"])
            fb.append(f)
        self.fast_blocks = fb

    def _tshift_fast(self, x, reverse=False):
        """x [T, V, C]: kênh c lấy từ khung t + o_c (o_c = c % k - k//2, đảo dấu nếu reverse); ngoài cửa sổ = 0."""
        k = self._k
        out = np.zeros_like(x)
        for g in range(k):
            o = g - k // 2
            if reverse:
                o = -o
            if o == 0:
                out[:, :, g::k] = x[:, :, g::k]
            elif o > 0:
                out[:-o, :, g::k] = x[o:, :, g::k]
            else:
                out[-o:, :, g::k] = x[:o, :, g::k]
        return out

    def logits(self, clip):
        """clip [T, V, 3] (kết quả normalize_clip) -> logits [num_class]. Bản nhanh (cùng kết quả logits_ref)."""
        if not hasattr(self, "fast_blocks"):
            self._prepare_fast()
        x = np.asarray(clip, dtype=np.float32) * self.f_data_scale + self.f_data_shift      # [T, V, C]
        V = self.V
        for f in self.fast_blocks:
            T = x.shape[0]
            ci, co = f["in"], f["out"]
            # SpatialShiftGCN
            g = x.reshape(T, V * ci)[:, f["shift_in"]].reshape(T * V, ci)
            g = (g.reshape(T, V, ci) * f["mask"]).reshape(T * V, ci) @ f["lin_w"] + f["lin_b"]
            g = g.reshape(T, V * co)[:, f["shift_out"]] * f["gs"] + f["gh"]
            g = g.reshape(T, V, co)
            if f["down"] is None:
                g = g + x
            else:
                W, b = f["down"]
                g = g + (x.reshape(T * V, ci) @ W + b).reshape(T, V, co)
            g = np.maximum(g, 0.0)
            # ShiftTCN
            y = self._tshift_fast(g * f["s1"] + f["h1"])
            y = np.maximum(y.reshape(T * V, co) @ f["t_w"] + f["t_b"], 0.0).reshape(T, V, co)
            y = self._tshift_fast(y, reverse=True)
            st = f["stride"]
            if st > 1:
                y = y[::st]
            y = y * f["s2"] + f["h2"]
            # residual
            if f["res"] == "id":
                y = y + x
            elif f["res"] is not None:
                W, b = f["res"]
                xs = x[::st] if st > 1 else x
                y = y + (xs.reshape(-1, ci) @ W + b).reshape(y.shape)
            x = np.maximum(y, 0.0)
        feat = x.mean(axis=(0, 1))
        return self.fc_w @ feat + self.fc_b

    def logits_ref(self, clip):
        """Bản gốc (bố cục [C, T, V], tensordot) — giữ để kiểm tra bản nhanh."""
        x = np.asarray(clip, dtype=np.float32).transpose(2, 0, 1)    # [C, T, V]
        x = x * self.data_scale + self.data_shift
        for blk in self.blocks:
            y = self._tcn(self._gcn(x, blk), blk)
            if blk["res"] == "id":
                y = y + x
            elif blk["res"] is not None:
                y = y + self._conv1x1(x[:, ::blk["stride"], :] if blk["stride"] > 1 else x, blk["res"])
            x = np.maximum(y, 0.0)
        feat = x.mean(axis=(1, 2))
        return self.fc_w @ feat + self.fc_b

    def predict(self, clip):
        """P(ngã) = softmax(logits)[1]."""
        z = self.logits(clip).astype(np.float64)
        z = z - z.max()
        e = np.exp(z)
        return float(e[1] / e.sum())
