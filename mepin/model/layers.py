import math

import torch
import torch.nn as nn
from torch_scatter import scatter_sum


# Modified from yang-song/score_sde_pytorch
class GaussianFourierBasis(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, num_basis: int):
        super().__init__()
        assert num_basis % 2 == 0
        self.num_basis = num_basis
        freqs = torch.randn(num_basis // 2) * 2 * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor):
        args = self.freqs * x[..., None]
        emb = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
        return emb


class BesselBasis(nn.Module):
    def __init__(self, num_basis: int, r_max: float, eps: float = 1e-6):
        super().__init__()
        self.num_basis = num_basis
        freqs = torch.arange(1, num_basis + 1, dtype=torch.float) * math.pi / r_max
        prefactor = torch.tensor(math.sqrt(2.0 / r_max), dtype=torch.float)
        self.register_buffer("freqs", freqs)
        self.register_buffer("prefactor", prefactor)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        args = self.freqs * x[..., None]
        x = x.clamp(min=self.eps)
        rbf = self.prefactor * torch.sin(args) / x[..., None]
        return rbf


class CosineCutoff(nn.Module):
    def __init__(self, r_max: float):
        super().__init__()
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float))

    def forward(self, x: torch.Tensor):
        x_cut = 0.5 * (1.0 + torch.cos(x * math.pi / self.r_max))
        x_cut = x_cut * (x < self.r_max).float()
        return x_cut


class MessageBlock(nn.Module):
    def __init__(self, num_features: int, num_radial_basis: int):
        super().__init__()
        self.num_features = num_features

        self.mlp_phi = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )
        self.linear_W = nn.Linear(num_radial_basis, num_features * 3)

    def forward(
        self,
        s: torch.Tensor,  # [n_nodes, 1, n_feats]
        v: torch.Tensor,  # [n_nodes, 3, n_feats]
        radial_embeddings: torch.Tensor,  # [n_edges, 1, num_radial_basis]
        f_cut: torch.Tensor,  # [n_edges, 1]
        unit_vectors: torch.Tensor,  # [n_edges, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
    ):
        idx_i, idx_j = edge_index[0], edge_index[1]
        n_nodes = s.shape[0]
        phi = self.mlp_phi(s)
        W = self.linear_W(radial_embeddings) * f_cut[..., None]
        x = phi[idx_j] * W
        x_s, x_vv, x_vs = torch.split(x, self.num_features, dim=-1)
        ds = scatter_sum(x_s, idx_i, dim=0, dim_size=n_nodes)
        x_v = v[idx_j] * x_vv + x_vs * unit_vectors[..., None]
        dv = scatter_sum(x_v, idx_i, dim=0, dim_size=n_nodes)
        return s + ds, v + dv


class DualMessageBlock(nn.Module):
    def __init__(self, num_features: int, num_radial_basis: int):
        super().__init__()
        self.num_features = num_features

        self.mlp_phi = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 4),
        )
        self.linear_W = nn.Linear(num_radial_basis, num_features * 4)

    def forward(
        self,
        s: torch.Tensor,  # [n_nodes, 1, n_feats]
        v: torch.Tensor,  # [n_nodes, 3, n_feats]
        radial_embeddings_1: torch.Tensor,  # [n_edges, 1, num_radial_basis]
        radial_embeddings_2: torch.Tensor,  # [n_edges, 1, num_radial_basis]
        f_cut_1: torch.Tensor,  # [n_edges, 1]
        f_cut_2: torch.Tensor,  # [n_edges, 1]
        unit_vectors_1: torch.Tensor,  # [n_edges, 3]
        unit_vectors_2: torch.Tensor,  # [n_edges, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
    ):
        idx_i, idx_j = edge_index[0], edge_index[1]
        n_nodes = s.shape[0]
        phi = self.mlp_phi(s)
        W = (
            self.linear_W(radial_embeddings_1) * f_cut_1[..., None]
            + self.linear_W(radial_embeddings_2) * f_cut_2[..., None]
        )
        x = phi[idx_j] * W
        x_s, x_vv, x_vs_1, x_vs_2 = torch.split(x, self.num_features, dim=-1)
        ds = scatter_sum(x_s, idx_i, dim=0, dim_size=n_nodes)
        x_v = (
            v[idx_j] * x_vv
            + x_vs_1 * unit_vectors_1[..., None]
            + x_vs_2 * unit_vectors_2[..., None]
        )
        dv = scatter_sum(x_v, idx_i, dim=0, dim_size=n_nodes)
        return s + ds, v + dv


class DualCrossMessageBlock(nn.Module):
    def __init__(self, num_features: int, num_radial_basis: int):
        super().__init__()
        self.num_features = num_features

        self.mlp_phi = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 6),
        )
        self.linear_W = nn.Linear(num_radial_basis, num_features * 6)

    def forward(
        self,
        s: torch.Tensor,  # [n_nodes, 1, n_feats]
        v: torch.Tensor,  # [n_nodes, 3, n_feats]
        radial_embeddings_1: torch.Tensor,  # [n_edges, 1, num_radial_basis]
        radial_embeddings_2: torch.Tensor,  # [n_edges, 1, num_radial_basis]
        f_cut_1: torch.Tensor,  # [n_edges, 1]
        f_cut_2: torch.Tensor,  # [n_edges, 1]
        unit_vectors_1: torch.Tensor,  # [n_edges, 3]
        unit_vectors_2: torch.Tensor,  # [n_edges, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
    ):
        idx_i, idx_j = edge_index[0], edge_index[1]
        n_nodes = s.shape[0]
        phi = self.mlp_phi(s)
        W = (
            self.linear_W(radial_embeddings_1) * f_cut_1[..., None]
            + self.linear_W(radial_embeddings_2) * f_cut_2[..., None]
        )
        x = phi[idx_j] * W
        x_s, x_vv, x_vs_1, x_vs_2, x_vc_1, x_vc_2 = torch.split(
            x, self.num_features, dim=-1
        )
        ds = scatter_sum(x_s, idx_i, dim=0, dim_size=n_nodes)
        x_v = (
            v[idx_j] * x_vv
            + x_vs_1 * unit_vectors_1[..., None]
            + x_vs_2 * unit_vectors_2[..., None]
            # Cross terms
            + x_vc_1 * torch.cross(v[idx_j], unit_vectors_1[..., None], dim=1)
            + x_vc_2 * torch.cross(v[idx_j], unit_vectors_2[..., None], dim=1)
        )
        dv = scatter_sum(x_v, idx_i, dim=0, dim_size=n_nodes)
        return s + ds, v + dv


class TripleCrossMessageBlock(nn.Module):
    def __init__(self, num_features: int, num_radial_basis: int):
        super().__init__()
        self.num_features = num_features

        self.mlp_phi = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 8),
        )
        self.linear_W = nn.Linear(num_radial_basis, num_features * 8)

    def forward(
        self,
        s: torch.Tensor,  # [n_nodes, 1, n_feats]
        v: torch.Tensor,  # [n_nodes, 3, n_feats]
        radial_embeddings_1: torch.Tensor,  # [n_edges, 1, num_radial_basis]
        radial_embeddings_2: torch.Tensor,  # [n_edges, 1, num_radial_basis]
        radial_embeddings_3: torch.Tensor,  # [n_edges, 1, num_radial_basis]
        f_cut_1: torch.Tensor,  # [n_edges, 1]
        f_cut_2: torch.Tensor,  # [n_edges, 1]
        f_cut_3: torch.Tensor,  # [n_edges, 1]
        unit_vectors_1: torch.Tensor,  # [n_edges, 3]
        unit_vectors_2: torch.Tensor,  # [n_edges, 3]
        unit_vectors_3: torch.Tensor,  # [n_edges, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
    ):
        idx_i, idx_j = edge_index[0], edge_index[1]
        n_nodes = s.shape[0]
        phi = self.mlp_phi(s)
        W = (
            self.linear_W(radial_embeddings_1) * f_cut_1[..., None]
            + self.linear_W(radial_embeddings_2) * f_cut_2[..., None]
            + self.linear_W(radial_embeddings_3) * f_cut_3[..., None]
        )
        x = phi[idx_j] * W
        x_s, x_vv, x_vs_1, x_vs_2, x_vs_3, x_vc_1, x_vc_2, x_vc_3 = torch.split(
            x, self.num_features, dim=-1
        )
        ds = scatter_sum(x_s, idx_i, dim=0, dim_size=n_nodes)
        x_v = (
            v[idx_j] * x_vv
            + x_vs_1 * unit_vectors_1[..., None]
            + x_vs_2 * unit_vectors_2[..., None]
            + x_vs_3 * unit_vectors_3[..., None]
            # Cross terms
            + x_vc_1 * torch.cross(v[idx_j], unit_vectors_1[..., None], dim=1)
            + x_vc_2 * torch.cross(v[idx_j], unit_vectors_2[..., None], dim=1)
            + x_vc_3 * torch.cross(v[idx_j], unit_vectors_3[..., None], dim=1)
        )
        dv = scatter_sum(x_v, idx_i, dim=0, dim_size=n_nodes)
        return s + ds, v + dv


class UpdateBlock(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.num_features = num_features
        self.mlp_a = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )
        self.linear_UV = nn.Linear(num_features, num_features * 2, bias=False)
        self.eps = eps

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        U_v, V_v = torch.split(self.linear_UV(v), self.num_features, dim=-1)
        V_v_norm = torch.sqrt((V_v**2).sum(dim=-2, keepdim=True) + self.eps)
        a = self.mlp_a(torch.cat((s, V_v_norm), dim=-1))
        a_vv, a_sv, a_ss = torch.split(a, self.num_features, dim=-1)
        dv = a_vv * U_v
        ds = a_ss + a_sv * torch.sum(U_v * V_v, dim=-2, keepdim=True)
        return s + ds, v + dv


class GatedEquivariantBlock(nn.Module):
    """Modified gated equivariant block to output a single vector.
    See PaiNN paper Fig. 3 or schnetpack.nn.equivariant module"""

    def __init__(
        self,
        num_scalar_inputs: int,
        num_vector_inputs: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_scalar_inputs = num_scalar_inputs
        self.num_vector_inputs = num_vector_inputs
        self.linear_v = nn.Linear(num_vector_inputs, 2, bias=False)
        self.mlp_s = nn.Sequential(
            nn.Linear(num_scalar_inputs + 1, num_scalar_inputs + 1),
            nn.SiLU(),
            nn.Linear(num_scalar_inputs + 1, 1),
        )
        self.eps = eps

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        W_v1, W_v2 = torch.split(self.linear_v(v), 1, dim=-1)
        W_v2_norm = torch.sqrt((W_v2**2).sum(dim=-2, keepdim=True) + self.eps)
        s_out = self.mlp_s(torch.cat((s, W_v2_norm), dim=-1))
        v_out = W_v1 * s_out
        return v_out


class GatedEquivariantScalarBlock(nn.Module):
    """Modified gated equivariant block to output a single scalar.
    See PaiNN paper Fig. 3 or schnetpack.nn.equivariant module"""

    def __init__(
        self,
        num_scalar_inputs: int,
        num_vector_inputs: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_scalar_inputs = num_scalar_inputs
        self.num_vector_inputs = num_vector_inputs
        self.linear_v = nn.Linear(num_vector_inputs, 1, bias=False)
        self.mlp_s = nn.Sequential(
            nn.Linear(num_scalar_inputs + 1, num_scalar_inputs + 1),
            nn.SiLU(),
            nn.Linear(num_scalar_inputs + 1, 1),
        )
        self.eps = eps

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        W_v = self.linear_v(v)
        W_v_norm = torch.sqrt((W_v**2).sum(dim=-2, keepdim=True) + self.eps)
        s_out = self.mlp_s(torch.cat((s, W_v_norm), dim=-1))
        return s_out
