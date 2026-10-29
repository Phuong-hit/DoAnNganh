
"""
model_stgcn.py
--------------
ST-GCN cho phân loại chuỗi skeleton:
0 = không ngã
1 = ngã

Đầu vào:
    x: [N, C, T, V]
    N: batch size
    C: số kênh (3)
    T: số frame
    V: số keypoint (17)

Đầu ra:
    logits: [N, 2]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .classifier import register_model
from .graph import build_adjacency


def to_model_input(X):
    """Chuyển numpy [N,T,V,C] thành tensor float32 [N,C,T,V]."""
    x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    return x.permute(0, 3, 1, 2).contiguous()


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        temporal_kernel=9,
        stride=1,
        dropout=0.2,
        residual=True,
    ):
        super().__init__()

        self.num_partitions = A.shape[0]

        # A cố định, không được cập nhật bằng gradient.
        self.register_buffer("A", A.clone())

        # Trọng số mức độ quan trọng của từng cạnh.
        self.edge_importance = nn.Parameter(torch.ones_like(A))

        # Graph convolution: tạo đặc trưng riêng cho từng nhóm cạnh.
        self.gcn = nn.Conv2d(
            in_channels,
            out_channels * self.num_partitions,
            kernel_size=1,
        )

        padding = (temporal_kernel - 1) // 2

        # Temporal convolution: học chuyển động qua các frame.
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [N,C,T,V]
        n, _, t, v = x.shape

        y = self.gcn(x)
        y = y.reshape(
            n, self.num_partitions, -1, t, v
        )

        # Kết hợp thông tin từ các nút lân cận.
        A = self.A * self.edge_importance
        y = torch.einsum("nkctv,kvw->nctw", y, A)

        y = self.tcn(y)

        if self.residual is not None:
            y = y + self.residual(x)

        return self.relu(y)


@register_model("stgcn")
class STGCN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=2,
        channels=(64, 64, 64, 128, 128, 128, 256, 256, 256),
        temporal_kernel=9,
        dropout=0.2,
        A=None,
    ):
        super().__init__()

        if A is None:
            A = torch.from_numpy(build_adjacency())
        else:
            A = torch.as_tensor(A, dtype=torch.float32)

        if A.ndim != 3 or A.shape[1] != A.shape[2]:
            raise ValueError("A phải có dạng [K,V,V].")

        self.num_nodes = A.shape[1]
        self.in_channels = in_channels

        # Chuẩn hóa đặc trưng theo từng kênh và từng keypoint.
        self.data_bn = nn.BatchNorm1d(in_channels * self.num_nodes)

        strides = [1, 1, 1, 2, 1, 1, 2, 1, 1]
        if len(channels) != len(strides):
            raise ValueError("Số lớp channels không hợp lệ.")

        blocks = []
        previous = in_channels

        for i, current in enumerate(channels):
            blocks.append(
                STGCNBlock(
                    in_channels=previous,
                    out_channels=current,
                    A=A,
                    temporal_kernel=temporal_kernel,
                    stride=strides[i],
                    dropout=dropout,
                    residual=(i != 0),
                )
            )
            previous = current

        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(previous, num_classes)

    def forward(self, x):
        # x: [N,C,T,V]
        if x.ndim != 4:
            raise ValueError("Đầu vào phải có dạng [N,C,T,V].")

        n, c, t, v = x.shape

        if c != self.in_channels or v != self.num_nodes:
            raise ValueError(
                f"Cần C={self.in_channels}, V={self.num_nodes}; "
                f"nhận được C={c}, V={v}."
            )

        # [N,C,T,V] -> [N,V*C,T]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(n, v * c, t)
        x = self.data_bn(x)

        # [N,V*C,T] -> [N,C,T,V]
        x = x.reshape(n, v, c, t)
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.blocks(x)

        # Global average pooling trên thời gian và keypoint.
        x = x.mean(dim=(2, 3))

        return self.fc(x)
