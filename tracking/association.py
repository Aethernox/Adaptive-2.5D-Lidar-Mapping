"""
Data Association using Hungarian Algorithm and Distance Gating
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def associate_detections_to_trackers(
    detections: np.ndarray,      # Shape (M, 7): [x, y, z, l, w, h, heading]
    trackers: np.ndarray,        # Shape (N, 7): [x, y, z, l, w, h, heading]
    distance_threshold: float = 2.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assigns detections to tracked object bounding boxes using Euclidean centroid distance.
    
    Returns:
        matches: (K, 2) array of [det_idx, track_idx]
        unmatched_detections: array of det_idx
        unmatched_trackers: array of track_idx
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int)
        )
    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            np.arange(len(trackers))
        )

    # Compute pairwise 3D centroid distance matrix
    det_centers = detections[:, :3] # (M, 3)
    trk_centers = trackers[:, :3]   # (N, 3)
    
    # Distance matrix (M, N)
    diff = det_centers[:, np.newaxis, :] - trk_centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    matches = []
    unmatched_detections = set(range(len(detections)))
    unmatched_trackers = set(range(len(trackers)))

    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] <= distance_threshold:
            matches.append([r, c])
            unmatched_detections.discard(r)
            unmatched_trackers.discard(c)

    matches_arr = np.array(matches, dtype=int) if len(matches) > 0 else np.empty((0, 2), dtype=int)
    return (
        matches_arr,
        np.array(list(unmatched_detections), dtype=int),
        np.array(list(unmatched_trackers), dtype=int)
    )
