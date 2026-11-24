KEYPOINT_NAMES = [
    "nose",             # 0
    "left_eye",         # 1
    "right_eye",        # 2
    "left_ear",         # 3
    "right_ear",        # 4
    "left_shoulder",    # 5
    "right_shoulder",   # 6
    "left_elbow",       # 7
    "right_elbow",      # 8
    "left_wrist",       # 9
    "right_wrist",      # 10
    "left_hip",         # 11
    "right_hip",        # 12
    "left_knee",        # 13
    "right_knee",       # 14
    "left_ankle",       # 15
    "right_ankle",      # 16
]

NUM_KEYPOINTS = 17


KEYPOINT_INDEX = {
    name: index
    for index, name in enumerate(KEYPOINT_NAMES)
}


SKELETON_EDGE_NAMES = [
    
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),

    
    ("left_shoulder", "right_shoulder"),

    
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),

    
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]




BONE_NAMES = [
    
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),

    
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
]




def _to_indices(pairs):
    return [
        (KEYPOINT_INDEX[a], KEYPOINT_INDEX[b])
        for a, b in pairs
    ]


SKELETON_EDGES = _to_indices(SKELETON_EDGE_NAMES)
BONES = _to_indices(BONE_NAMES)




HEAD = [
    KEYPOINT_INDEX["nose"],
    KEYPOINT_INDEX["left_eye"],
    KEYPOINT_INDEX["right_eye"],
    KEYPOINT_INDEX["left_ear"],
    KEYPOINT_INDEX["right_ear"],
]

SHOULDERS = [
    KEYPOINT_INDEX["left_shoulder"],
    KEYPOINT_INDEX["right_shoulder"],
]

HIPS = [
    KEYPOINT_INDEX["left_hip"],
    KEYPOINT_INDEX["right_hip"],
]

KNEES = [
    KEYPOINT_INDEX["left_knee"],
    KEYPOINT_INDEX["right_knee"],
]

ANKLES = [
    KEYPOINT_INDEX["left_ankle"],
    KEYPOINT_INDEX["right_ankle"],
]

TORSO = SHOULDERS + HIPS




FLIP_INDEX = [
    0,
    2, 1,
    4, 3,
    6, 5,
    8, 7,
    10, 9,
    12, 11,
    14, 13,
    16, 15,
]




assert NUM_KEYPOINTS == 17

assert len(KEYPOINT_NAMES) == NUM_KEYPOINTS

assert len(set(KEYPOINT_NAMES)) == NUM_KEYPOINTS

assert len(FLIP_INDEX) == NUM_KEYPOINTS

assert sorted(FLIP_INDEX) == list(range(NUM_KEYPOINTS))

assert len(SKELETON_EDGES) == len(SKELETON_EDGE_NAMES)

assert len(BONES) == len(BONE_NAMES)
