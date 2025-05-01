import ase
import ase.io
import numpy as np
import torch
from torch_geometric.data import Batch, Data

from mepin.tools.geometry import get_neighbor_list_batch, kabsch_align


def create_reaction_batch(
    reactant_atoms: ase.Atoms,
    product_atoms: ase.Atoms,
    interp_traj: list[ase.Atoms] | None = None,
    edge_cutoff: float = 6.0,
    num_images: int = 10,
    use_geodesic: bool = False,
) -> Batch:
    """Create a input batch for the reaction path model given the reactant and
    product geometries."""

    # Read the reactant and product geometries
    atomic_numbers = torch.tensor(reactant_atoms.get_atomic_numbers(), dtype=torch.long)
    reactant_positions = reactant_atoms.get_positions().astype(np.float32)
    product_positions = product_atoms.get_positions().astype(np.float32)

    # Align the reactant and product geometries
    product_positions = kabsch_align(product_positions, reactant_positions)

    # Load the geodesic interpolation
    if use_geodesic:
        control_positions = [frame.get_positions() for frame in interp_traj]
        control_positions = kabsch_align(
            np.array(control_positions), reactant_positions
        ).astype(np.float32)

    reactant_positions = torch.from_numpy(reactant_positions)
    product_positions = torch.from_numpy(product_positions)

    # Construct the edge index
    edge_index = get_neighbor_list_batch(
        positions_batch=[reactant_positions, product_positions],
        cutoff=edge_cutoff,
        lattice=None,
        periodic=False,
    )
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Interpolation progress
    time = torch.linspace(0, 1, num_images, dtype=torch.float32)

    data_list = [
        Data(
            atomic_numbers=atomic_numbers,
            num_nodes=atomic_numbers.size(0),
            reactant_positions=reactant_positions,
            product_positions=product_positions,
            edge_index=edge_index,
            graph_time=t,
        )
        for t in time
    ]
    batch = Batch.from_data_list(data_list)
    batch["reaction_index"] = torch.tensor([0], dtype=torch.long)
    if use_geodesic:
        batch["control_positions"] = torch.from_numpy(control_positions)
    return batch
