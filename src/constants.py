"""
constants.py
------------
Nơi DUY NHẤT định nghĩa keypoint và cách nối khớp. Các tệp khác chỉ import
từ đây, không tự khai báo lại.

Thứ tự 17 keypoint theo chuẩn COCO (khớp với đầu ra của YOLO-pose).
Chỉ số trong PoseResult.joints[i] ứng với KEYPOINT_NAMES[i].
"""

KEYPOINT_NAMES = [
    "nose",                          # 0
    "left_eye", "right_eye",         # 1, 2
    "left_ear", "right_ear",         # 3, 4
    "left_shoulder", "right_shoulder",  # 5, 6
    "left_elbow", "right_elbow",     # 7, 8
    "left_wrist", "right_wrist",     # 9, 10
    "left_hip", "right_hip",         # 11, 12
    "left_knee", "right_knee",       # 13, 14
    "left_ankle", "right_ankle",     # 15, 16
]
NUM_KEYPOINTS = len(KEYPOINT_NAMES)

# Tra chỉ số theo tên: KEYPOINT_INDEX["left_hip"] -> 11
KEYPOINT_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# ---------------------------------------------------------------------------
# Cạnh để VẼ skeleton (dạng tên, dễ đọc / dễ sửa).
# ---------------------------------------------------------------------------
SKELETON_EDGE_NAMES = [
    # đầu
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"),
    # vai
    ("left_shoulder", "right_shoulder"),
    # tay
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    # thân
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # chân
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

# ---------------------------------------------------------------------------
# "Xương" dùng để KIỂM TRA độ dài trong SkeletonFilter (chỉ tay, chân, thân;
# không gồm đầu và các cạnh ngang vì độ dài hay thay đổi theo góc nhìn).
# ---------------------------------------------------------------------------
BONE_NAMES = [
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
]


def _to_indices(pairs):
    return [(KEYPOINT_INDEX[a], KEYPOINT_INDEX[b]) for a, b in pairs]


# Dạng chỉ số (0-16), dùng trong code xử lý mảng numpy.
SKELETON_EDGES = _to_indices(SKELETON_EDGE_NAMES)
BONES = _to_indices(BONE_NAMES)

# ---------------------------------------------------------------------------
# Nhóm khớp thường dùng khi tính đặc trưng ngã (thêm/bớt tùy nhóm).
# ---------------------------------------------------------------------------
HEAD = [KEYPOINT_INDEX[n] for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")]
SHOULDERS = [KEYPOINT_INDEX["left_shoulder"], KEYPOINT_INDEX["right_shoulder"]]
HIPS = [KEYPOINT_INDEX["left_hip"], KEYPOINT_INDEX["right_hip"]]
KNEES = [KEYPOINT_INDEX["left_knee"], KEYPOINT_INDEX["right_knee"]]
ANKLES = [KEYPOINT_INDEX["left_ankle"], KEYPOINT_INDEX["right_ankle"]]
TORSO = SHOULDERS + HIPS

# Kiểm tra nhất quán ngay khi import: sai là báo lỗi sớm.
assert NUM_KEYPOINTS == 17
assert len(set(KEYPOINT_NAMES)) == NUM_KEYPOINTS, "Tên keypoint bị trùng"

# Hoán đổi trái/phải khi lật ngang (augmentation)
FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
assert sorted(FLIP_INDEX) == list(range(NUM_KEYPOINTS))