import abc
import warnings
from typing import Any, Literal

import ase.data
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_scatter import scatter_sum

from mepin.model.potentials import ConvergenceError, InvalidPositionsError
from mepin.tools.geometry import get_unit_vectors_and_lengths


class BaseLoss(nn.Module, abc.ABC):
    """Base class for the loss functions."""

    def __init__(
        self,
        weight: float = 1.0,
        derivative_order: Literal[0, 1, 2] = 0,
        log_clip_value: float | None = None,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.derivative_order = derivative_order
        self.log_clip_value = log_clip_value

    @abc.abstractmethod
    def forward(self, **kwargs: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError


class GeodesicLoss(BaseLoss):
    """Compute the loss for minimizing the geodesic distance."""

    derivative_order = 1

    def __init__(
        self,
        alpha: float = 1.7,
        beta: float = 0.01,
        distance_clamp: float = 0.1,
        weight: float = 1.0,
        log_clip_value: float = 100.0,
    ) -> None:
        super().__init__(weight, self.derivative_order, log_clip_value)
        self.alpha = alpha
        self.beta = beta
        self.distance_clamp = distance_clamp
        self.register_buffer(
            "covalent_radii", torch.tensor(ase.data.covalent_radii, dtype=torch.float)
        )

    def get_coord(self, batch: Batch, x_t: torch.Tensor) -> torch.Tensor:
        # TODO: Need to check when the pbcs are used
        # Compute the distance
        _, dist = get_unit_vectors_and_lengths(x_t, batch.edge_index, shifts=None)
        dist = dist.squeeze()  # [num_edges, 1] -> [num_edges,]
        dist_ref = self.covalent_radii[batch["atomic_numbers"][batch.edge_index]].sum(
            dim=0
        )

        # Compute internal coordinates
        # Clamp the distance for long-range part to avoid numerical instability
        internal_coords = torch.exp(
            -self.alpha * (dist - dist_ref) / dist_ref
        ) + self.beta * dist_ref / torch.clamp(
            dist, min=self.distance_clamp
        )  # [num_edges,]
        return internal_coords

    def forward(
        self,
        batch: Batch,
        results: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x_t, dx_dt = results["x_t"], results["dx_dt"]  # [num_nodes, 3]
        # TODO: think whether we need to recompute edge_index
        # batch = base.refeaturize_atomgraphs(batch, x_t, recompute_neighbors=True)

        # Compute d/dt(internal coordinates)
        def wrap_coord(x_t: torch.Tensor) -> torch.Tensor:
            return self.get_coord(batch, x_t)

        _, jvp = torch.func.jvp(wrap_coord, (x_t,), (dx_dt,))  # [num_edges,]
        metric_energy = scatter_sum(
            src=jvp.pow(2),
            index=batch.batch[batch.edge_index[0]],  # graph index for each edge
            dim=0,
            dim_size=batch.batch[-1] + 1,
        )
        loss = metric_energy.mean()
        return loss


class MaxFluxLoss(BaseLoss):
    """Compute the loss for maximizing the reactive flux."""

    derivative_order = 1

    def __init__(
        self,
        energy_model: nn.Module,
        beta: float = 20.0,
        use_velocity_norm: bool = True,
        weight: float = 1.0,
        log_clip_value: float = 0.0,
    ) -> None:
        super().__init__(weight, self.derivative_order, log_clip_value)
        self.energy_model = energy_model
        self.beta = beta
        self.use_velocity_norm = use_velocity_norm

    def forward(self, batch: Batch, results: dict[str, torch.Tensor]) -> torch.Tensor:
        x_t, dx_dt = results["x_t"], results["dx_dt"]  # [num_nodes, 3]
        try:
            energies = self.energy_model(batch=batch, new_positions=x_t)
        except ConvergenceError as e:
            # If the energy model fails to converge, return a detached log_clip_value
            # because the positions themselves are not the problem
            warnings.warn(
                f"Energy model failed to converge with error: {e}. "
                f"Returning a detached log_clip_value."
            )
            return torch.tensor(self.log_clip_value, dtype=x_t.dtype, device=x_t.device)
        except (InvalidPositionsError, RuntimeError) as e:
            # If the energy model fails due to invalid positions, raise the error
            # because positions are the problem (hence the model cannot be trained)
            raise e
        lse_args = self.beta * energies
        if self.use_velocity_norm:
            dx_dt_norm = scatter_sum(
                src=dx_dt.pow(2).sum(dim=1),
                index=batch.batch,
                dim=0,
                dim_size=batch.batch[-1] + 1,
            ).sqrt()  # [num_graphs,]
            lse_args = lse_args + dx_dt_norm.log()
        batch_size = batch["reaction_index"].shape[0]
        lse_args = lse_args.reshape(batch_size, -1)  # [batch_size, num_images]
        loss = torch.logsumexp(lse_args, dim=1).mean() / self.beta
        return loss


class ArcLengthLoss(BaseLoss):
    """Compute the loss for minimizing the arc length."""

    derivative_order = 2

    def __init__(
        self,
        weight: float = 1.0,
        log_clip_value: float = 500.0,
    ) -> None:
        super().__init__(weight, self.derivative_order, log_clip_value)

    def forward(self, batch: Batch, results: dict[str, torch.Tensor]) -> torch.Tensor:
        dx_dt, d2x_dt2 = results["dx_dt"], results["d2x_dt2"]  # [num_nodes, 3]
        args = ((dx_dt * d2x_dt2) ** 2).sum(dim=1)  # [num_nodes,]
        loss_graph = scatter_sum(
            src=args,
            index=batch.batch,
            dim=0,
            dim_size=batch.batch[-1] + 1,
        )  # [num_graphs,]
        loss = loss_graph.mean()
        return loss
