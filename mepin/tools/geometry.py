import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from vesin import NeighborList


class MorseCoordinates(nn.Module):
    def __init__(
        self, alpha: float = 1.7, beta: float = 0.01, clamp: float | None = 0.01
    ):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float))
        if clamp is not None:
            self.register_buffer("clamp", torch.tensor(clamp, dtype=torch.float))

    def forward(
        self,
        path: torch.Tensor,  # [n_images, n_atoms, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
        ref_dist: torch.Tensor,  # [n_edges]
        shifts: torch.Tensor | None = None,  # [n_atoms, 3]
    ) -> torch.Tensor:
        """Compute the internal coordinates of the path."""
        # Compute distances
        src, dst = edge_index
        disp = path[:, src, :] - path[:, dst, :]
        if shifts is not None:
            disp = disp + shifts[None, :, :]
        dist = torch.norm(disp, dim=-1)
        if hasattr(self, "clamp"):
            dist = torch.clamp(dist, min=self.clamp)

        # Compute internal coordinates
        return (
            torch.exp(-self.alpha * (dist - ref_dist) / ref_dist)
            + self.beta * ref_dist / dist
        )  # [n_images, n_edges]

    def geodesic_length(
        self,
        path: torch.Tensor,  # [n_images, n_atoms, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
        ref_dist: torch.Tensor,  # [n_edges]
        shifts: torch.Tensor | None = None,  # [n_atoms, 3]
    ) -> torch.Tensor:
        """Compute the approximate geodesic length of the path."""
        path_mid = 0.5 * (path[:-1] + path[1:])
        coord = self.forward(path, edge_index, ref_dist, shifts)  # [n_images, n_edges]
        coord_mid = self.forward(path_mid, edge_index, ref_dist, shifts)
        return (
            torch.norm(coord[:-1] - coord_mid, dim=-1).sum()
            + torch.norm(coord_mid - coord[1:], dim=-1).sum()
        )


def get_unit_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor | None = None,  # [n_edges, 3]
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_i, idx_j = edge_index[0], edge_index[1]  # [n_edges]
    if shifts is not None:
        vectors = positions[idx_j] - positions[idx_i] + shifts  # [n_edges, 3]
    else:
        vectors = positions[idx_j] - positions[idx_i]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    unit_vectors = vectors / (lengths + eps)
    return unit_vectors, lengths


def get_neighbor_list_batch(
    positions_batch: np.ndarray | list[np.ndarray],  # [n_configs, n_atoms, 3]
    cutoff: float = 5.0,
    lattice: np.ndarray | None = None,  # [3, 3]
    periodic: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Construct neighbor list for a batch of coordinates, as a union of all
    neighbor lists for each configuration in the batch. The neighbor list is
    represented as a tuple of edge index and shifts (if periodic)."""

    nl = NeighborList(cutoff=cutoff, full_list=True)
    edge_attrs_all = []
    for positions in positions_batch:
        if periodic:
            src, dst, shifts = nl.compute(
                quantities="ijS",
                points=positions,
                box=lattice,
                periodic=True,
            )
            edge_attrs = np.hstack((src[:, None], dst[:, None], shifts)).astype(int)
        else:
            src, dst = nl.compute(
                quantities="ij",
                points=positions,
                box=np.zeros((3, 3)),  # non-periodic
                periodic=False,
            )
            edge_attrs = np.hstack((src[:, None], dst[:, None])).astype(int)
        edge_attrs_all.append(edge_attrs)
    edge_attrs = np.unique(np.vstack(edge_attrs_all), axis=0)
    edge_index = edge_attrs[:, :2].T
    if periodic:
        shifts = edge_attrs[:, 2:] @ lattice  # convert shifts to Cartesian
        return edge_index, shifts
    else:
        return edge_index


def kabsch(coords: np.ndarray, ref_coords: np.ndarray) -> np.ndarray:
    """Computes the optimal rotation matrix using the Kabsch algorithm.
    Assumes that the point clouds are centered at the origin."""
    # SVD of the covariance matrix
    U, _, Vh = np.linalg.svd(coords.swapaxes(-2, -1) @ ref_coords)

    # Flip if the orthogonal matrices contain reflections
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vh))
    Vh[..., -1] *= d[..., None]

    # Compute the rotation matrix and return it
    return U @ Vh


def kabsch_align(coords: np.ndarray, ref_coords: np.ndarray) -> np.ndarray:
    """Aligns two point clouds using the Kabsch algorithm."""
    # Center the point clouds
    ref_center = ref_coords.mean(axis=0)
    ref_coords = ref_coords - ref_center
    coords = coords - coords.mean(axis=-2, keepdims=True)

    # Compute the optimal rotation matrix
    R = kabsch(coords, ref_coords)

    # Apply the transformation and return the aligned coordinates
    return coords @ R + ref_center


def random_rotation_matrix(rng: np.random.Generator, num_samples: int) -> np.ndarray:
    """Generate a random orthogonal matrix using QR decomposition.
    Ref: https://math.stackexchange.com/a/1602779"""
    z = rng.normal(loc=0.0, scale=1.0, size=(num_samples, 3, 3))
    q, r = np.linalg.qr(z)
    sign = 2 * (np.diagonal(r, axis1=-2, axis2=-1) >= 0) - 1
    rot = q
    rot *= sign[..., None, :]
    rot[:, 0, :] *= np.linalg.det(rot)[..., None]
    return rot.astype(np.float32)


def random_small_rotation_matrix(
    rng: np.random.Generator, scale: float = 0.1
) -> np.ndarray:
    """Generate small rotation matrices with random yaw, pitch, and roll angles."""
    yaw, pitch, roll = rng.normal(loc=0.0, scale=scale, size=3)
    return R.from_euler("zyx", [yaw, pitch, roll]).as_matrix().astype(np.float32)


def get_pca_rotation(points: np.ndarray) -> np.ndarray:
    """Get the PCA rotation matrix for a set of points.
    Apply to points by multiplying the output on the right (@ rot)."""
    # Compute the PCA eigenvectors
    points = points - points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvectors by decreasing eigenvalues, flip if necessary
    eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1
    return eigenvectors.astype(np.float32)


def catmull_rom_spline(
    control_points: torch.Tensor,  # [num_controls, num_points, 3]
    time: torch.Tensor,  # [batch_size,]
) -> torch.Tensor:  # [batch_size, num_points, 3]
    # Map global time to segment index and local parameter u in [0, 1]
    N = control_points.shape[0]
    scaled_t = time * (N - 1)
    seg_idx = scaled_t.floor().long()
    u = (scaled_t - seg_idx.float())[:, None, None]  # [batch_size, 1, 1]

    # Get the four control points for the spline segment
    p0 = control_points[(seg_idx - 1).clamp(0, N - 1)]
    p1 = control_points[seg_idx]
    p2 = control_points[(seg_idx + 1).clamp(0, N - 1)]
    p3 = control_points[(seg_idx + 2).clamp(0, N - 1)]

    # Compute the Catmull-Rom spline
    return 0.5 * (
        (-p0 + 3 * p1 - 3 * p2 + p3) * u**3
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
        + (-p0 + p2) * u
        + 2 * p1
    )


def cubic_b_spline(
    control_points: torch.Tensor,  # [num_controls, num_points, 3]
    time: torch.Tensor,  # [batch_size,]
) -> torch.Tensor:  # [batch_size, num_points, 3]
    # Map global time to segment index and local parameter u in [0, 1]
    N = control_points.shape[0]
    scaled_t = time * (N + 1)
    seg_idx = scaled_t.floor().long()
    u = (scaled_t - seg_idx.float())[:, None, None]  # [batch_size, 1, 1]

    # Get the four control points for the spline segment
    # seg_idx = 0: (0, 0, 0, 1), ..., seg_idx = N: (N-2, N-1, N-1, N-1)
    p0 = control_points[(seg_idx - 2).clamp(0, N - 1)]
    p1 = control_points[(seg_idx - 1).clamp(0, N - 1)]
    p2 = control_points[seg_idx.clamp(0, N - 1)]
    p3 = control_points[(seg_idx + 1).clamp(0, N - 1)]

    # Compute the cubic B-spline
    return (
        (-p0 + 3 * p1 - 3 * p2 + p3) * u**3
        + (3 * p0 - 6 * p1 + 3 * p2) * u**2
        + (-3 * p0 + 3 * p2) * u
        + (p0 + 4 * p1 + p2)
    ) / 6
