"""
model_adapters.py
-----------------
Nối các model không viết theo hợp đồng của src/classifier.py vào FallClassifier, để web / main.py / evaluate.py
dùng chung mà KHÔNG phải sửa file gốc của model:

  shiftgcn   src/model_shiftgcn.py (giữ nguyên bản của nhóm)
  b0_rule    luật B0 (src/b0_rule.py), checkpoint là file JSON tham số (models/b0_rule_v1.json)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .b0_rule import B0Rule
from .classifier import register_model
from .model_shiftgcn import Model


@register_model("shiftgcn")
class ShiftGCNClassifier(Model):
    """build_model("shiftgcn", in_channels=3, num_classes=2, num_point=17, num_frames=17).
    Tên tham số giữ nguyên Model, nên nạp thẳng được shiftgcn_best.pt."""

    def __init__(self, in_channels=3, num_classes=2, num_point=17, num_frames=17, **kwargs):
        super().__init__(num_class=num_classes, num_point=num_point, in_channels=in_channels,
                         num_frames=num_frames, **kwargs)


@register_model("b0_rule")
class B0RuleModule(nn.Module):
    """Luật B0 dưới dạng [N,C,T,V] -> logits [N,2]. Quyết định vẫn do luật: P(ngã) >= 0.5 đúng khi luật báo ngã;
    độ lớn của P lấy từ score() của luật, chỉ để hiển thị."""

    def __init__(self, in_channels=3, num_classes=2, params=None):
        super().__init__()
        self.rule = B0Rule.from_params(params or {})

    def forward(self, x):
        X = x.permute(0, 2, 3, 1).detach().cpu().numpy()
        pred, score = self.rule.predict(X), self.rule.score(X)
        p = np.clip(np.where(pred == 1, 0.5 + 0.5 * score, 0.49 * score), 1e-6, 1 - 1e-6)
        p = torch.as_tensor(p, dtype=torch.float32, device=x.device)
        return torch.log(torch.stack([1 - p, p], dim=1))
