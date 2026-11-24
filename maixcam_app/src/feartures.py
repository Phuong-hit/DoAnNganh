import numpy as np

from .constants import BONES, FLIP_INDEX, HIPS


def _mean_xy(points):
    """Tính trung bình tọa độ các điểm [N, 2]."""
    count = len(points)

    if count == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    sx = 0.0
    sy = 0.0

    for i in range(count):
        sx += float(points[i][0])
        sy += float(points[i][1])

    return np.array(
        [sx / count, sy / count],
        dtype=np.float32
    )


def _median(values):
    """Median bằng Python, không phụ thuộc np.median."""
    if len(values) == 0:
        return 0.0

    values = sorted(values)
    n = len(values)

    if n % 2 == 1:
        return float(values[n // 2])

    return float(
        (values[n // 2 - 1] + values[n // 2]) / 2.0
    )


def _distance(p1, p2):
    """Khoảng cách Euclidean giữa hai điểm."""
    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])

    return (dx * dx + dy * dy) ** 0.5


def normalize_clip(seq, mask, aspect=1.0):


    seq = np.asarray(seq, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    W = seq.shape[0]

    out = np.zeros((W, 17, 3), dtype=np.float32)

    
    has_valid = False

    for t in range(W):
        for j in range(17):
            if mask[t, j]:
                has_valid = True
                break

        if has_valid:
            break

    if not has_valid:
        return out

    

    xy = np.zeros((W, 17, 2), dtype=np.float32)

    for t in range(W):
        for j in range(17):
            if mask[t, j]:
                xy[t, j, 0] = seq[t, j, 0] * aspect
                xy[t, j, 1] = seq[t, j, 1]

    

    hip_points = []

    for t in range(W):
        for hip_idx in HIPS:
            if mask[t, hip_idx]:
                hip_points.append(xy[t, hip_idx])

    if len(hip_points) > 0:
        center = _mean_xy(hip_points)

    else:
        valid_points = []

        for t in range(W):
            for j in range(17):
                if mask[t, j]:
                    valid_points.append(xy[t, j])

        center = _mean_xy(valid_points)

   

    bone_lengths = []

    for a, b in BONES:

        for t in range(W):

            if mask[t, a] and mask[t, b]:

                length = _distance(
                    xy[t, a],
                    xy[t, b]
                )

                bone_lengths.append(length)

    scale = _median(bone_lengths)

    if scale < 1e-3:
        scale = 1.0

    

    for t in range(W):

        for j in range(17):

            if mask[t, j]:

                out[t, j, 0] = (
                    float(xy[t, j, 0]) - float(center[0])
                ) / scale

                out[t, j, 1] = (
                    float(xy[t, j, 1]) - float(center[1])
                ) / scale

                out[t, j, 2] = 1.0

            else:

                out[t, j, 0] = 0.0
                out[t, j, 1] = 0.0
                out[t, j, 2] = 0.0

    return out


def normalize_batch(X, M, aspect=1.0):
    """
    Chuẩn hóa batch.

    X: [N, W, 17, 3]
    M: [N, W, 17]

    Output: [N, W, 17, 3]

    
    """

    X = np.asarray(X, dtype=np.float32)
    M = np.asarray(M, dtype=bool)

    N = X.shape[0]
    W = X.shape[1]

    out = np.zeros(
        (N, W, 17, 3),
        dtype=np.float32
    )

    for n in range(N):

        out[n] = normalize_clip(
            X[n],
            M[n],
            aspect
        )

    return out


def flip_clip(clip):
   

    clip = np.asarray(clip, dtype=np.float32)

    W = clip.shape[0]

    out = np.zeros(
        (W, 17, 3),
        dtype=np.float32
    )

    for t in range(W):

        for j in range(17):

            src_j = FLIP_INDEX[j]

            # Hoán đổi keypoint trái/phải
            out[t, j, 0] = -clip[t, src_j, 0]
            out[t, j, 1] = clip[t, src_j, 1]
            out[t, j, 2] = clip[t, src_j, 2]

            # Khớp không quan sát được giữ bằng 0
            if out[t, j, 2] == 0.0:
                out[t, j, 0] = 0.0
                out[t, j, 1] = 0.0

    return out
