"""Shift-GCN viết lại cho NPU (MaixCAM / TPU-MLIR): CÙNG phép tính với src/model_shiftgcn.py, nhưng chỉ dùng
Conv2d / BatchNorm / ReLU / Reshape / Permute / Mean / Linear (không có index_select, không có dịch theo chỉ số).

  SpatialShiftGCN:  dịch(V*C) -> nhân mặt nạ -> Linear(C_in->C_out) chung cho mọi khớp -> dịch ngược -> BN1d(V*C_out)
                    = MỘT phép tuyến tính cố định V*C_in -> V*C_out mỗi khung  =>  Conv 1x1 trên tensor [N, V*C, T, 1]
  TemporalShift:    kênh c lấy từ khung t + offset[c] (ngoài cửa sổ = 0)
                    = Conv depthwise kernel (k,1), trọng số one-hot, padding k//2

Dùng:
    from src.shiftgcn_npu import to_npu_model
    npu = to_npu_model(clf.model)          # clf = FallClassifier(...) (model gốc đã nạp trọng số)
    logits = npu(x)                        # x [N, 3, T, V], giống model gốc (sai số ~1e-5)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .model_shiftgcn import Model, ShiftBlock, SpatialShiftGCN, TemporalShift


def _bn_affine(bn: nn.BatchNorm1d | nn.BatchNorm2d):
    scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    return scale, bn.bias.detach() - bn.running_mean * scale


def _spatial_matrix(g: SpatialShiftGCN) -> tuple[torch.Tensor, torch.Tensor]:
    """Ma trận M [V*out, V*in] và bias [V*out] sao cho  y_flat = M @ x_flat + b  (thứ tự phẳng: khớp, rồi kênh),
    gồm dịch vào, mặt nạ, Linear, dịch ra và BN1d. Dựng bằng cách cho đi qua các vector cơ sở (chính xác tuyệt đối)."""
    V, cin, cout = g.num_nodes, g.in_channels, g.out_channels
    with torch.no_grad():
        def core(x_flat):                                    # x_flat [B, V*cin] -> [B, V*cout] (chưa cộng down)
            b = x_flat.shape[0]
            x = x_flat.index_select(1, g.shift_in).view(b, V, cin)
            x = x * (torch.tanh(g.Feature_Mask) + 1.0)
            x = torch.matmul(x, g.Linear_weight) + g.Linear_bias
            x = x.reshape(b, -1).index_select(1, g.shift_out)
            return x
        bias_pre = core(torch.zeros(1, V * cin, dtype=torch.float64).to(g.Linear_weight.dtype))[0]
        cols = core(torch.eye(V * cin, dtype=g.Linear_weight.dtype)) - bias_pre        # [V*cin, V*cout]
        s, h = _bn_affine(g.bn)
        M = cols.T * s[:, None]
        b = bias_pre * s + h
    return M, b


def _temporal_conv(ts: TemporalShift, kernel: int) -> nn.Conv2d:
    """Conv depthwise (k,1) one-hot tái tạo TemporalShift (cùng zero-padding ở 2 đầu cửa sổ)."""
    C, T = ts.channels, ts.num_frames
    idx = ts.idx.view(C, T)
    c = torch.arange(C)[:, None]
    t = torch.arange(T)[None, :]
    offs = (idx - (c * T + t))                               # độ dịch của từng kênh (ô hợp lệ)
    valid = ts.valid.view(C, T) > 0
    off_c = torch.zeros(C, dtype=torch.long)
    for ch in range(C):
        v = offs[ch][valid[ch]]
        off_c[ch] = int(v[0]) if len(v) else 0
    conv = nn.Conv2d(C, C, (kernel, 1), padding=(kernel // 2, 0), groups=C, bias=False)
    w = torch.zeros(C, 1, kernel, 1)
    w[torch.arange(C), 0, off_c + kernel // 2, 0] = 1.0       # out[t] = in[t + off]  (tương quan chéo)
    conv.weight.data.copy_(w)
    conv.weight.requires_grad_(False)
    return conv


class _GCN(nn.Module):
    def __init__(self, g: SpatialShiftGCN):
        super().__init__()
        self.V, self.cin, self.cout = g.num_nodes, g.in_channels, g.out_channels
        M, b = _spatial_matrix(g)
        self.lin = nn.Conv2d(self.V * self.cin, self.V * self.cout, 1)
        self.lin.weight.data.copy_(M[:, :, None, None])
        self.lin.bias.data.copy_(b)
        self.down = g.down                                    # Identity hoặc Conv1x1 + BN2d (giữ nguyên)

    def forward(self, x0):                                    # [N, C, T, V]
        n, c, t, v = x0.shape
        x = x0.permute(0, 3, 1, 2).reshape(n, v * c, t, 1)   # [N, V*C, T, 1], thứ tự kênh: khớp rồi kênh
        x = self.lin(x)                                       # [N, V*out, T, 1]
        x = x.reshape(n, v, self.cout, t).permute(0, 2, 3, 1)  # [N, out, T, V]
        return torch.relu(x + self.down(x0))


class _GCNUnfused(nn.Module):
    """Bản KHÔNG gộp (để thí nghiệm lượng tử hóa): dịch khớp bằng Gather (index_select), nhân mặt nạ, Linear chung
    C_in->C_out, Gather dịch ngược, BN1d tách riêng — đúng cấu trúc model gốc, chỉ BN viết thành nhân + cộng."""

    def __init__(self, g: SpatialShiftGCN):
        super().__init__()
        self.V, self.cin, self.cout = g.num_nodes, g.in_channels, g.out_channels
        with torch.no_grad():
            self.register_buffer("shift_in", g.shift_in.clone())
            self.register_buffer("shift_out", g.shift_out.clone())
            self.register_buffer("mask", (torch.tanh(g.Feature_Mask) + 1.0).reshape(1, 1, self.V, self.cin).clone())
            self.register_buffer("lin_w", g.Linear_weight.detach().clone())                 # [in, out]
            self.register_buffer("lin_b", g.Linear_bias.detach().reshape(1, 1, 1, self.cout).clone())
            s, h = _bn_affine(g.bn)
            self.register_buffer("bn_s", s.reshape(1, 1, -1).clone())
            self.register_buffer("bn_h", h.reshape(1, 1, -1).clone())
        self.down = g.down

    def forward(self, x0):                                    # [N, C, T, V]
        n, c, t, v = x0.shape
        x = x0.permute(0, 2, 3, 1).reshape(n, t, v * c)
        x = torch.index_select(x, 2, self.shift_in).reshape(n, t, v, c)
        x = torch.matmul(x * self.mask, self.lin_w) + self.lin_b           # [N, T, V, out]
        x = torch.index_select(x.reshape(n, t, v * self.cout), 2, self.shift_out)
        x = (x * self.bn_s + self.bn_h).reshape(n, t, v, self.cout).permute(0, 3, 1, 2)
        return torch.relu(x + self.down(x0))


def _perm_conv(idx: torch.Tensor) -> nn.Conv2d:
    """Conv 1x1 trọng số 0/1 làm phép dịch y[k] = x[idx[k]] (thay Gather mà cv181x không hỗ trợ)."""
    n = len(idx)
    conv = nn.Conv2d(n, n, 1, bias=False)
    w = torch.zeros(n, n, 1, 1)
    w[torch.arange(n), idx.long(), 0, 0] = 1.0
    conv.weight.data.copy_(w)
    conv.weight.requires_grad_(False)
    return conv


class _GCNPerm(nn.Module):
    """Bản KHÔNG gộp chạy được trên cv181x (thí nghiệm lượng tử hóa): dịch = Conv1x1 hoán vị 0/1 (INT8 biểu diễn
    chính xác), mặt nạ / Linear chung C_in->C_out / BN để RIÊNG từng phép như model gốc."""

    def __init__(self, g: SpatialShiftGCN):
        super().__init__()
        V, cin, cout = g.num_nodes, g.in_channels, g.out_channels
        self.V, self.cin, self.cout = V, cin, cout
        self.p_in, self.p_out = _perm_conv(g.shift_in), _perm_conv(g.shift_out)
        with torch.no_grad():
            self.register_buffer("mask", (torch.tanh(g.Feature_Mask) + 1.0).reshape(1, V, cin, 1).clone())
            self.lin = nn.Conv2d(cin, cout, 1)
            self.lin.weight.data.copy_(g.Linear_weight.detach().T[:, :, None, None])
            self.lin.bias.data.copy_(g.Linear_bias.detach().reshape(-1))
            s, h = _bn_affine(g.bn)
            self.register_buffer("bn_s", s.reshape(1, -1, 1, 1).clone())
            self.register_buffer("bn_h", h.reshape(1, -1, 1, 1).clone())
        self.down = g.down

    def forward(self, x0):                                    # [N, C, T, V]
        n, c, t, v = x0.shape
        x = self.p_in(x0.permute(0, 3, 1, 2).reshape(n, v * c, t, 1))           # [N, V*C, T, 1]
        x = x.reshape(n, v, c, t) * self.mask
        x = self.lin(x.permute(0, 2, 1, 3))                                    # [N, out, V, T]
        x = x.permute(0, 2, 1, 3).reshape(n, v * self.cout, t, 1)
        x = self.p_out(x) * self.bn_s + self.bn_h
        x = x.reshape(n, v, self.cout, t).permute(0, 2, 3, 1)                  # [N, out, T, V]
        return torch.relu(x + self.down(x0))


_GCN_KINDS = {"fused": _GCN, "gather": _GCNUnfused, "perm": _GCNPerm}


class _TCN(nn.Module):
    def __init__(self, tcn, kernel: int):
        super().__init__()
        self.bn = tcn.bn
        self.shift_in = _temporal_conv(tcn.shift_in, kernel)
        self.lin = tcn.temporal_linear
        self.shift_out = _temporal_conv(tcn.shift_out, kernel)
        self.stride = tcn.stride
        self.bn2 = tcn.bn2

    def forward(self, x):
        x = self.shift_in(self.bn(x))
        x = self.shift_out(torch.relu(self.lin(x)))
        if self.stride > 1:
            x = x[:, :, :: self.stride, :]
        return self.bn2(x)


class _Block(nn.Module):
    def __init__(self, b: ShiftBlock, kernel: int, gcn: str = "fused"):
        super().__init__()
        self.gcn = _GCN_KINDS[gcn](b.gcn)
        self.tcn = _TCN(b.tcn, kernel)
        self.residual = b.residual

    def forward(self, x):
        y = self.tcn(self.gcn(x))
        if self.residual is not None:
            y = y + self.residual(x)
        return torch.relu(y)


class ShiftGCNNPU(nn.Module):
    """[N, 3, T, V] -> logits [N, num_class]."""

    def __init__(self, m: Model, kernel: int = 3, gcn: str = "fused"):
        super().__init__()
        V, C = m.num_point, m.in_channels
        s, h = _bn_affine(m.data_bn)                          # BN1d(V*C), chỉ số v*C + c
        self.register_buffer("data_scale", s.view(V, C).T.reshape(1, C, 1, V).clone())
        self.register_buffer("data_shift", h.view(V, C).T.reshape(1, C, 1, V).clone())
        self.blocks = nn.Sequential(*[_Block(b, kernel, gcn) for b in m.blocks])
        self.fc = m.fc

    def forward(self, x):
        x = x * self.data_scale + self.data_shift
        x = self.blocks(x)
        return self.fc(x.mean(dim=(2, 3)))


def temporal_kernel(m: Model) -> int:
    ts = m.blocks[0].tcn.shift_in
    T = ts.num_frames
    idx = ts.idx.view(-1, T)
    c = torch.arange(idx.shape[0])[:, None]
    t = torch.arange(T)[None, :]
    off = (idx - (c * T + t)).abs()[ts.valid.view(-1, T) > 0]
    return int(2 * off.max() + 1) if off.numel() else 1


def to_npu_model(m: Model, gcn: str = "fused") -> ShiftGCNNPU:
    """gcn: "fused" = GCN gộp thành MỘT Conv1x1 dày (bản dùng trên MaixCAM);
    "perm" = không gộp, dịch bằng Conv1x1 hoán vị 0/1; "gather" = không gộp, dịch bằng Gather (cv181x KHÔNG hỗ trợ)."""
    m = m.eval().cpu().float()
    return ShiftGCNNPU(m, temporal_kernel(m), gcn).eval()


def max_abs_diff(m: Model, npu: ShiftGCNNPU, clips: np.ndarray) -> float:
    x = torch.as_tensor(np.asarray(clips, dtype=np.float32)).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        return float((m.eval()(x) - npu(x)).abs().max())
