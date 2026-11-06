"""
model_lstm.py
-------------
LSTM phân loại chuỗi skeleton (B1), cùng hợp đồng với ST-GCN để so sánh công bằng:

    đầu vào  x [N, C, T, V]  (C=3: x, y đã chuẩn hóa bằng features.normalize_clip + mask; T=20; V=17)
    đầu ra   logits [N, 2]   (0 = KHÔNG NGÃ, 1 = NGÃ)

Khác ST-GCN: LSTM không biết cấu trúc khung xương (đồ thị khớp). Mỗi khung được duỗi thành một
vector V*C số, rồi LSTM đọc lần lượt T khung để học chuyển động theo thời gian.

Kiến trúc (mặc định ~0.3 triệu tham số, ST-GCN ~3 triệu):
    [N,C,T,V] -> [N,T,V*C]
      (+ vận tốc dx, dy giữa hai khung liên tiếp nếu velocity=True; chỉ tính khi khớp
         quan sát được ở cả hai khung, còn lại = 0)
    -> BatchNorm theo đặc trưng (giống data_bn của ST-GCN)
    -> Linear(embed) + ReLU + Dropout                   (nén 17 khớp thành một vector khung)
    -> LSTM(hidden, layers, bidirectional tùy chọn)
    -> gộp [trung bình theo thời gian, trạng thái khung cuối]
    -> Dropout -> Linear -> logits [N, 2]

Vận tốc được tính BÊN TRONG model từ chính đầu vào, nên dataset, web và FallClassifier không
phải thay đổi gì. Các tham số kiến trúc được lưu trong checkpoint ("model_kwargs").
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .classifier import register_model


@register_model("lstm")
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        num_nodes: int = 17,
        embed: int = 128,
        hidden: int = 128,
        layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.3,
        velocity: bool = True,
    ):
        super().__init__()
        if in_channels < 3:
            raise ValueError("Cần in_channels >= 3 (x, y, mask).")
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.velocity = velocity

        feat = num_nodes * (in_channels + (2 if velocity else 0))
        self.input_bn = nn.BatchNorm1d(feat)
        self.embed = nn.Sequential(
            nn.Linear(feat, embed),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=embed,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out * 2, num_classes))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """[N,C,T,V] -> [N,T,F] với F = V*C (+ V*2 nếu có vận tốc)."""
        n, c, t, v = x.shape
        x = x.permute(0, 2, 3, 1)                         # [N,T,V,C]
        parts = [x]
        if self.velocity:
            xy, m = x[..., :2], x[..., 2:3]               # mask = kênh 3 (0/1)
            vel = torch.zeros_like(xy)
            both = m[:, 1:] * m[:, :-1]                    # chỉ khi thấy khớp ở cả hai khung
            vel[:, 1:] = (xy[:, 1:] - xy[:, :-1]) * both
            parts.append(vel)
        x = torch.cat(parts, dim=-1)                       # [N,T,V,C(+2)]
        return x.reshape(n, t, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Đầu vào phải có dạng [N,C,T,V].")
        n, c, t, v = x.shape
        if c != self.in_channels or v != self.num_nodes:
            raise ValueError(f"Cần C={self.in_channels}, V={self.num_nodes}; nhận được C={c}, V={v}.")

        f = self._features(x)                              # [N,T,F]
        f = self.input_bn(f.transpose(1, 2)).transpose(1, 2)
        f = self.embed(f)                                  # [N,T,embed]
        y, _ = self.lstm(f)                                # [N,T,out]
        pooled = torch.cat([y.mean(dim=1), y[:, -1]], dim=1)
        return self.head(pooled)
