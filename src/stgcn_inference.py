"""
stgcn_inference.py
------------------
Giữ để code cũ vẫn chạy: `from src.stgcn_inference import STGCNInference`.

Logic đã chuyển sang src/classifier.py (FallClassifier), dùng chung cho ST-GCN, LSTM, ...
Kiến trúc lấy từ khóa "arch" trong checkpoint; checkpoint cũ không có khóa này là ST-GCN.

Nhãn: 0 = KHÔNG NGÃ, 1 = NGÃ. Mặc định cửa sổ 20 khung, trượt 5 khung.
"""
from .classifier import FallClassifier


class STGCNInference(FallClassifier):
    def __init__(self, checkpoint_path, window_size=20, stride_frames=5, device=None,
                 fall_threshold=0.5):
        super().__init__(checkpoint_path, window_size=window_size, stride_frames=stride_frames,
                         device=device, fall_threshold=fall_threshold)
