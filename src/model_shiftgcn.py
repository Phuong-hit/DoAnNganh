"""
model_shiftgcn.py
-----------------
Shift-GCN (Cheng và cộng sự, CVPR 2020) viết lại bằng PyTorch THUẦN để chạy được
trên máy bình thường (CPU hoặc GPU), cho dữ liệu của dự án:

    đầu vào  x : [N, C, T, V]  = [batch, 3 kênh (x, y, mask), 10 khung, 17 khớp]
    đầu ra     : [N, num_class] (logits; 0 = không ngã, 1 = ngã)

Khác bản gốc (kchengiva/Shift-GCN):
  * Không dùng nhân CUDA `cuda.shift`. Phép dịch theo THỜI GIAN viết bằng
    index_select với chỉ số cố định (loại "naive": độ dịch cố định cho từng kênh).
  * Không có `.cuda()` cứng, không cần đồ thị (bản non-local không dùng ma trận kề).
  * Số khớp là tham số (17), không còn số 25 cứng; không có chiều "người" M.
  * Mạng nhỏ mặc định (4 khối) vì dữ liệu ít.
  * Không còn np.int và các hàm init cũ.

Dùng:
    from src.model_shiftgcn import Model
    model = Model(num_class=2, num_point=17, num_frames=10)
    logits = model(x)                      # x: [N, 3, 10, 17]

Chạy thử:  python model_shiftgcn.py
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Bảng chỉ số (thuần numpy, kiểm tra riêng được)
# ----------------------------------------------------------------------
def spatial_shift_index(num_nodes, channels, inverse=False):
    """Chỉ số dịch theo KHÔNG GIAN trên vector phẳng độ dài nút*kênh (thứ tự: nút, rồi kênh).

    Chiều thuận : kênh j của nút i lấy từ nút (i + j) mod V.
    Chiều ngược : kênh j của nút i lấy từ nút (i - j) mod V (đảo lại chiều thuận).
    """
    size = num_nodes * channels
    sign = -1 if inverse else 1
    idx = np.empty(size, dtype=np.int64)
    for i in range(num_nodes):
        for j in range(channels):
            idx[i * channels + j] = (i * channels + j + sign * j * channels) % size
    return idx


def temporal_shift_tables(channels, num_frames, kernel=3, reverse=False):
    """Bảng dịch theo THỜI GIAN, trên vector phẳng độ dài kênh*khung (thứ tự: kênh, rồi khung).

    Kênh c được dịch (c mod kernel) - kernel//2 khung (ví dụ kernel=3: -1, 0, +1, -1, ...).
    Ô lấy từ ngoài cửa sổ thì bằng 0.
    Trả (idx [C*T] kiểu int64, valid [C*T] kiểu float32 gồm 0/1).
    """
    offsets = (np.arange(channels) % kernel) - kernel // 2
    if reverse:
        offsets = -offsets
    idx = np.zeros((channels, num_frames), dtype=np.int64)
    valid = np.zeros((channels, num_frames), dtype=np.float32)
    for c in range(channels):
        for t in range(num_frames):
            src = t + int(offsets[c])
            if 0 <= src < num_frames:
                idx[c, t] = c * num_frames + src
                valid[c, t] = 1.0
            else:
                idx[c, t] = c * num_frames + t     # chỉ số hợp lệ bất kỳ, sẽ bị nhân 0
    return idx.reshape(-1), valid.reshape(-1)


# ----------------------------------------------------------------------
# Các khối
# ----------------------------------------------------------------------
class SpatialShiftGCN(nn.Module):
    """Shift graph convolution theo không gian (non-local): dịch -> mặt nạ -> 1x1 -> dịch ngược."""

    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = nn.Identity()

        self.Linear_weight = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.normal_(self.Linear_weight, 0.0, math.sqrt(1.0 / out_channels))
        self.Linear_bias = nn.Parameter(torch.zeros(1, 1, out_channels))
        self.Feature_Mask = nn.Parameter(torch.zeros(1, num_nodes, in_channels))

        self.bn = nn.BatchNorm1d(num_nodes * out_channels)

        self.register_buffer(
            "shift_in",
            torch.from_numpy(spatial_shift_index(num_nodes, in_channels)),
            persistent=False,
        )
        self.register_buffer(
            "shift_out",
            torch.from_numpy(spatial_shift_index(num_nodes, out_channels, inverse=True)),
            persistent=False,
        )

    def forward(self, x0):                       # x0: [N, C, T, V]
        n, c, t, v = x0.shape
        x = x0.permute(0, 2, 3, 1).contiguous().view(n * t, v * c)
        x = x.index_select(1, self.shift_in).view(n * t, v, c)
        x = x * (torch.tanh(self.Feature_Mask) + 1.0)

        x = torch.matmul(x, self.Linear_weight) + self.Linear_bias      # [nt, V, out]

        x = x.reshape(n * t, -1)
        x = x.index_select(1, self.shift_out)
        x = self.bn(x)
        x = x.view(n, t, v, self.out_channels).permute(0, 3, 1, 2)      # [N, out, T, V]

        x = x + self.down(x0)
        return F.relu(x)


class TemporalShift(nn.Module):
    """Dịch theo thời gian với độ dịch cố định cho từng kênh (thay cho nhân CUDA Shift)."""

    def __init__(self, channels, num_frames, kernel=3, reverse=False):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        idx, valid = temporal_shift_tables(channels, num_frames, kernel, reverse)
        self.register_buffer("idx", torch.from_numpy(idx), persistent=False)
        self.register_buffer("valid", torch.from_numpy(valid).view(1, -1, 1), persistent=False)

    def forward(self, x):                        # x: [N, C, T, V]
        n, c, t, v = x.shape
        if c != self.channels or t != self.num_frames:
            raise ValueError(
                f"TemporalShift cần C={self.channels}, T={self.num_frames}; nhận C={c}, T={t}."
            )
        x = x.reshape(n, c * t, v)
        x = x.index_select(1, self.idx) * self.valid
        return x.reshape(n, c, t, v)


class ShiftTCN(nn.Module):
    """Shift temporal convolution: BN -> dịch -> 1x1 -> ReLU -> dịch ngược -> BN (có stride theo thời gian)."""

    def __init__(self, in_channels, out_channels, num_frames, kernel=3, stride=1):
        super().__init__()
        self.stride = stride
        self.bn = nn.BatchNorm2d(in_channels)
        self.shift_in = TemporalShift(in_channels, num_frames, kernel)
        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal_(self.temporal_linear.weight, mode="fan_out")
        nn.init.constant_(self.temporal_linear.bias, 0.0)
        self.shift_out = TemporalShift(out_channels, num_frames, kernel, reverse=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.shift_in(x)
        x = F.relu(self.temporal_linear(x))
        x = self.shift_out(x)
        if self.stride > 1:
            x = x[:, :, :: self.stride, :]
        return self.bn2(x)


class ShiftBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, num_frames,
                 kernel=3, stride=1, residual=True):
        super().__init__()
        self.gcn = SpatialShiftGCN(in_channels, out_channels, num_nodes)
        self.tcn = ShiftTCN(out_channels, out_channels, num_frames, kernel, stride)
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        y = self.tcn(self.gcn(x))
        if self.residual is not None:
            y = y + self.residual(x)
        return F.relu(y)


class Model(nn.Module):
    """Shift-GCN nhỏ: [N, C, T, V] -> [N, num_class]."""

    def __init__(self, num_class=2, num_point=17, in_channels=3, num_frames=17,
                 channels=(32, 32, 64, 64), strides=None, temporal_kernel=3):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        strides = tuple(strides) if strides is not None else (1,) * len(channels)
        if len(strides) != len(channels):
            raise ValueError("strides phải có cùng độ dài với channels.")

        self.num_class = num_class
        self.num_point = num_point
        self.in_channels = in_channels
        self.num_frames = num_frames

        self.data_bn = nn.BatchNorm1d(in_channels * num_point)

        blocks = []
        prev, t = in_channels, num_frames
        for i, (ch, st) in enumerate(zip(channels, strides)):
            blocks.append(ShiftBlock(prev, ch, num_point, t, temporal_kernel, st,
                                     residual=(i != 0)))
            prev = ch
            t = (t - 1) // st + 1                 # số khung sau stride
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(prev, num_class)

    def forward(self, x):                         # x: [N, C, T, V]
        if x.ndim != 4:
            raise ValueError("Đầu vào phải có dạng [N, C, T, V].")
        n, c, t, v = x.shape
        if c != self.in_channels or v != self.num_point or t != self.num_frames:
            raise ValueError(
                f"Cần C={self.in_channels}, T={self.num_frames}, V={self.num_point}; "
                f"nhận C={c}, T={t}, V={v}."
            )

        # BatchNorm theo từng (khớp, kênh), giống bản gốc khi M = 1
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1)             # [N, C, T, V]

        x = self.blocks(x)
        x = x.mean(dim=(2, 3))                                 # trung bình theo thời gian và khớp
        return self.fc(x)


# ----------------------------------------------------------------------
# Đăng ký vào src/classifier.py (FallClassifier, evaluate.py, test_tool.py, web dùng chung)
# ----------------------------------------------------------------------
def _register():
    try:
        from .classifier import register_model
    except ImportError:                       # chạy trực tiếp: python src/model_shiftgcn.py
        return

    @register_model("shiftgcn")
    class ShiftGCNClassifier(Model):
        """Cùng hợp đồng với ST-GCN / LSTM: build_model("shiftgcn", in_channels=3, num_classes=2,
        num_point=17, num_frames=17, ...) -> [N, C, T, V] -> logits [N, 2].
        Tên tham số trong state_dict giữ nguyên Model, nên nạp thẳng được shiftgcn_best.pt."""

        def __init__(self, in_channels=3, num_classes=2, num_point=17, num_frames=17, **kwargs):
            super().__init__(num_class=num_classes, num_point=num_point, in_channels=in_channels,
                             num_frames=num_frames, **kwargs)


_register()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# Chạy thử
# ----------------------------------------------------------------------
def _self_test():
    torch.manual_seed(0)

    # 1) chỉ số dịch không gian: kênh j của nút i lấy từ nút (i + j) mod V, và dịch ngược đảo lại
    V, C = 17, 8
    a, b = spatial_shift_index(V, C), spatial_shift_index(V, C, inverse=True)
    for i in range(V):
        for j in range(C):
            assert a[i * C + j] == ((i + j) % V) * C + j
    assert (a[b] == np.arange(V * C)).all()
    print("[OK] chỉ số dịch không gian")

    # 2) dịch thời gian: out[c, t] = in[c, t + offset], ngoài cửa sổ = 0
    ts = TemporalShift(channels=3, num_frames=6, kernel=3)
    x = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1).repeat(1, 3, 1, 2)
    y = ts(x)
    assert y[0, 0, :, 0].tolist() == [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]    # offset -1
    assert y[0, 1, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]    # offset 0
    assert y[0, 2, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 0.0]    # offset +1
    print("[OK] dịch thời gian")

    # 3) mô hình: hình dạng, lan truyền ngược, không NaN
    model = Model(num_class=2, num_point=17, num_frames=17)
    print(f"Số tham số huấn luyện: {count_parameters(model):,}")
    model.train()
    x = torch.randn(8, 3, 17, 17)
    out = model(x)
    assert out.shape == (8, 2), out.shape
    loss = F.cross_entropy(out, torch.randint(0, 2, (8,)))
    loss.backward()
    assert torch.isfinite(loss).item()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"[OK] forward/backward, loss = {loss.item():.4f}")

    # 4) chế độ eval: cùng đầu vào cho cùng kết quả, batch = 1 chạy được
    model.eval()
    with torch.no_grad():
        o1, o2 = model(x[:1]), model(x[:1])
    assert torch.allclose(o1, o2)
    print("[OK] eval, batch = 1")

    # 5) chuyển lên GPU nếu có
    if torch.cuda.is_available():
        model = model.cuda()
        with torch.no_grad():
            model(x[:2].cuda())
        print("[OK] GPU")
    else:
        print("(không có GPU, bỏ qua)")
    print("Tất cả kiểm tra đạt.")


if __name__ == "__main__":
    _self_test()