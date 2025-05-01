import torch  # noqa: F401
import torch.nn as nn
from torch_geometric.data import Data

from mepin.model.layers import (
    BesselBasis,
    CosineCutoff,
    GatedEquivariantBlock,
    GaussianFourierBasis,
    TripleCrossMessageBlock,
    UpdateBlock,
)
from mepin.tools.geometry import get_unit_vectors_and_lengths


class TripleCrossPaiNN(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_radial_basis: int,
        num_layers: int,
        num_elements: int,
        r_max: float,
        r_offset: float = 0.0,
        use_vector_embedding: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_radial_basis = num_radial_basis
        self.num_layers = num_layers
        self.r_max = r_max
        self.r_offset = r_offset

        assert num_features % 2 == 0, "Number of features must be even"
        self.atom_embedding = nn.Embedding(num_elements, num_features, padding_idx=0)
        self.time_embedding = GaussianFourierBasis(num_basis=num_features)
        self.radial_embedding = BesselBasis(num_basis=num_radial_basis, r_max=r_max)
        self.cutoff_fn = CosineCutoff(r_max=r_max)
        self.use_vector_embedding = use_vector_embedding
        if use_vector_embedding:
            self.linear_v = nn.Linear(1, num_features, bias=False)

        messages, updates = [], []
        for _ in range(num_layers):
            messages.append(TripleCrossMessageBlock(num_features, num_radial_basis))
            updates.append(UpdateBlock(num_features))
        self.messages = nn.ModuleList(messages)
        self.updates = nn.ModuleList(updates)
        self.output_block = GatedEquivariantBlock(
            num_scalar_inputs=num_features,
            num_vector_inputs=num_features,
        )

    def forward(self, data: Data):
        unit_vectors_1, lengths_1 = get_unit_vectors_and_lengths(
            data["reactant_positions"], data["edge_index"], shifts=None
        )
        unit_vectors_2, lengths_2 = get_unit_vectors_and_lengths(
            data["product_positions"], data["edge_index"], shifts=None
        )
        unit_vectors_3, lengths_3 = get_unit_vectors_and_lengths(
            data["interp_positions"], data["edge_index"], shifts=None
        )

        # Compute radial basis functions
        lengths_1 = (lengths_1 + self.r_offset).clamp(max=self.r_max)
        lengths_2 = (lengths_2 + self.r_offset).clamp(max=self.r_max)
        lengths_3 = (lengths_3 + self.r_offset).clamp(max=self.r_max)
        radial_embeddings_1 = self.radial_embedding(lengths_1)
        radial_embeddings_2 = self.radial_embedding(lengths_2)
        radial_embeddings_3 = self.radial_embedding(lengths_3)
        f_cut_1 = self.cutoff_fn(lengths_1)
        f_cut_2 = self.cutoff_fn(lengths_2)
        f_cut_3 = self.cutoff_fn(lengths_3)

        # Compute initial scalar and vector features
        s = self.atom_embedding(data["atomic_numbers"]) + self.time_embedding(
            data["time"]
        )
        s = s[:, None, :]  # [n_nodes, 1, n_feats]
        if self.use_vector_embedding:
            positions_diff = data["product_positions"] - data["reactant_positions"]
            v = self.linear_v(positions_diff[..., None])  # [n_nodes, 3, n_feats]
        else:
            v = torch.zeros_like(s).repeat(1, 3, 1)  # [n_nodes, 3, n_feats]

        for message, update in zip(self.messages, self.updates):
            s, v = message(
                s,
                v,
                radial_embeddings_1,
                radial_embeddings_2,
                radial_embeddings_3,
                f_cut_1,
                f_cut_2,
                f_cut_3,
                unit_vectors_1,
                unit_vectors_2,
                unit_vectors_3,
                data["edge_index"],
            )
            s, v = update(s, v)
        v_out = self.output_block(s, v).squeeze(-1)
        return v_out
