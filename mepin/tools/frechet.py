from typing import Callable

import numpy as np

from mepin.tools.geometry import kabsch_align


def rmsd(p1: np.ndarray, p2: np.ndarray) -> np.float64:
    """Computes the root mean square deviation (RMSD) between two point clouds of
    shape [num_points, 3]."""
    p1_aligned = kabsch_align(p1, p2)
    return np.sqrt(np.sum((p1_aligned - p2) ** 2) / p1.shape[0])


def frechet_distance(
    path1: np.ndarray,
    path2: np.ndarray,
    dist: Callable[[np.ndarray, np.ndarray], np.float64] = rmsd,
) -> np.float64:
    """Computes the Frechet distance between two paths of point clouds of shape
    [num_images, num_points, 3]."""
    dist_matrix = np.full((path1.shape[0], path2.shape[0]), -1, dtype=np.float64)

    def _inner(i, j):  # dynamic programming
        if dist_matrix[i, j] > -1:
            return dist_matrix[i, j]
        elif i == 0 and j == 0:
            dist_matrix[i, j] = dist(path1[i], path2[j])
        elif i > 0 and j == 0:
            dist_matrix[i, j] = max(_inner(i - 1, 0), dist(path1[i], path2[j]))
        elif i == 0 and j > 0:
            dist_matrix[i, j] = max(_inner(0, j - 1), dist(path1[i], path2[j]))
        elif i > 0 and j > 0:
            dist_matrix[i, j] = max(
                min(_inner(i - 1, j), _inner(i - 1, j - 1), _inner(i, j - 1)),
                dist(path1[i], path2[j]),
            )
        else:
            dist_matrix[i, j] = np.inf
        return dist_matrix[i, j]

    return _inner(path1.shape[0] - 1, path2.shape[0] - 1)
