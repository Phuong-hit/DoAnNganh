
"""
graph.py
--------
Ma trận kề cho ST-GCN, dựa trên spatial configuration.

A: [K, V, V], K=3:
  A[0]: root - chính khớp và láng giềng cùng khoảng cách tới tâm
  A[1]: centripetal - nguồn gần tâm hơn đích
  A[2]: centrifugal - nguồn xa tâm hơn đích

Để tương thích với skeleton COCO hiện tại, tâm được xấp xỉ
bằng khoảng cách đồ thị tới hai khớp hông.
Đây là biến thể dùng tâm đồ thị cố định.
"""
from __future__ import annotations

import numpy as np

from .constants import HIPS, NUM_KEYPOINTS, SKELETON_EDGES


def hop_distance(num_nodes=NUM_KEYPOINTS, edges=SKELETON_EDGES):
    """Tính khoảng cách ngắn nhất giữa mọi cặp nút."""
    distance = np.full(
        (num_nodes, num_nodes), np.inf, dtype=np.float32
    )
    np.fill_diagonal(distance, 0.0)

    for a, b in edges:
        distance[a, b] = 1.0
        distance[b, a] = 1.0

    for k in range(num_nodes):
        distance = np.minimum(
            distance,
            distance[:, k:k + 1] + distance[k:k + 1, :]
        )

    return distance


def normalize_digraph(adjacency):
    """Chuẩn hóa đồ thị theo bậc cột."""
    degree = adjacency.sum(axis=0)
    inv_degree = np.zeros_like(degree, dtype=np.float32)

    valid = degree > 0
    inv_degree[valid] = 1.0 / degree[valid]

    return adjacency * inv_degree[np.newaxis, :]


def build_adjacency(
    edges=SKELETON_EDGES,
    num_nodes=NUM_KEYPOINTS,
    center=HIPS,
):
    """
    Trả ma trận A [3,V,V].

    adjacency[source, target] biểu thị quan hệ nguồn -> đích.
    Đồ thị được chuẩn hóa trước khi chia thành các partition.
    """
    hop = hop_distance(num_nodes, edges)

    # Ma trận gồm self-loop và cạnh xương hai chiều.
    adjacency = np.eye(num_nodes, dtype=np.float32)

    for a, b in edges:
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0

    # Chuẩn hóa toàn bộ đồ thị trước khi phân vùng.
    adjacency = normalize_digraph(adjacency)

    # Khoảng cách tới hông gần nhất, dùng làm tâm xấp xỉ.
    center_distance = hop[:, list(center)].min(axis=1)

    root = np.zeros_like(adjacency)
    centripetal = np.zeros_like(adjacency)
    centrifugal = np.zeros_like(adjacency)

    for target in range(num_nodes):
        for source in range(num_nodes):
            if hop[source, target] > 1:
                continue

            ds = center_distance[source]
            dt = center_distance[target]
            value = adjacency[source, target]

            if source == target or ds == dt:
                root[source, target] = value
            elif ds < dt:
                centripetal[source, target] = value
            else:
                centrifugal[source, target] = value

    A = np.stack([root, centripetal, centrifugal], axis=0)
    return A.astype(np.float32)


if __name__ == "__main__":
    A = build_adjacency()

    print("A shape:", A.shape)
    print("Finite:", np.isfinite(A).all())
    print("Non-negative:", (A >= 0).all())
    print("Partition sums:", A.sum(axis=(1, 2)))

    assert A.shape == (3, NUM_KEYPOINTS, NUM_KEYPOINTS)
    assert np.isfinite(A).all()
    assert (A >= 0).all()
