from typing import Any

import hydra
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch_geometric.data import Batch

from mepin.model.models import TripleCrossPaiNN
from mepin.tools.geometry import cubic_b_spline


class TripleCrossPaiNNModule(L.LightningModule):
    def __init__(
        self,
        num_features: int = 64,
        num_radial_basis: int = 20,
        num_layers: int = 3,
        num_elements: int = 118,
        r_max: float = 6.0,
        r_offset: float = 0.5,
        use_vector_embedding: bool = False,
        use_geodesic: bool = False,
        optimizer_config: DictConfig = {"_target_": "torch.optim.Adam", "lr": 3e-4},
        loss_configs: dict[str, DictConfig] = [],
    ) -> None:
        super().__init__()
        self.use_geodesic = use_geodesic
        self.optimizer_config = optimizer_config
        self.loss_configs = loss_configs
        self.save_hyperparameters()

        # Define the model
        self.model = TripleCrossPaiNN(
            num_features=num_features,
            num_radial_basis=num_radial_basis,
            num_layers=num_layers,
            num_elements=num_elements,
            r_max=r_max,
            r_offset=r_offset,
            use_vector_embedding=use_vector_embedding,
        )

    def setup(self, stage) -> None:
        # Initialize the loss modules only during the fit stage
        if stage == "fit":
            losses = {
                loss_name: hydra.utils.instantiate(loss_config)
                for loss_name, loss_config in self.loss_configs.items()
            }
            self.losses = nn.ModuleDict(losses)

            # Get max derivative order for the losses
            self.max_derivative_order = max(
                loss.derivative_order for loss in self.losses.values()
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Remove the loss modules from the state dict before saving
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if key.startswith("losses"):
                del state_dict[key]

    def forward(self, batch: Batch) -> torch.Tensor:
        batch["time"] = batch["graph_time"][batch.batch]
        _t = batch["time"][:, None]
        if self.use_geodesic:
            batch["interp_positions"] = cubic_b_spline(
                control_points=batch["control_positions"],
                time=batch["graph_time"],
            ).reshape(-1, 3)
        else:  # linear interpolation
            batch["interp_positions"] = (1 - _t) * batch[
                "reactant_positions"
            ] + _t * batch["product_positions"]
        return batch["interp_positions"] + _t * (1.0 - _t) * self.model(batch)

    def configure_optimizers(self) -> optim.Optimizer:
        return hydra.utils.instantiate(self.optimizer_config, self.model.parameters())

    @torch.enable_grad()
    def get_derivatives(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Compute the second, first and zeroth derivatives of the predicted
        transition path with respect to time for a batch of graphs.

        Note: This might not be the most efficient way to compute the
        derivatives, as memory requirements grow as O(B^2) where B is the
        batch size."""

        device = batch.batch.device
        node_index = torch.arange(batch.num_nodes, device=device)

        # Wrap the model to compute the second derivative
        def wrap_model(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal batch
            batch["time"] = t[batch.batch]
            _t = batch["time"][:, None]
            if self.use_geodesic:
                batch["interp_positions"] = cubic_b_spline(
                    control_points=batch["control_positions"],
                    time=t,
                ).reshape(-1, 3)
            else:  # linear interpolation
                batch["interp_positions"] = (1 - _t) * batch[
                    "reactant_positions"
                ] + _t * batch["product_positions"]
            x_t = batch["interp_positions"] + _t * (1.0 - _t) * self.model(batch)
            return x_t, x_t

        def wrap_dx_dt(
            t: torch.Tensor,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            dx_dt, x_t = torch.func.jacfwd(wrap_model, has_aux=True)(t)
            dx_dt = dx_dt[node_index, :, batch.batch]  # [num_nodes, 3]
            return dx_dt, (dx_dt, x_t)

        def wrap_d2x_dt2(
            t: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            d2x_dt2, (dx_dt, x_t) = torch.func.jacfwd(wrap_dx_dt, has_aux=True)(t)
            d2x_dt2 = d2x_dt2[node_index, :, batch.batch]  # [num_nodes, 3]
            return d2x_dt2, dx_dt, x_t

        # Compute the derivatives
        if self.max_derivative_order == 2:
            d2x_dt2, dx_dt, x_t = wrap_d2x_dt2(batch["graph_time"])
            return {"d2x_dt2": d2x_dt2, "dx_dt": dx_dt, "x_t": x_t}
        elif self.max_derivative_order == 1:
            _, (dx_dt, x_t) = wrap_dx_dt(batch["graph_time"])
            return {"dx_dt": dx_dt, "x_t": x_t}
        elif self.max_derivative_order == 0:
            x_t, _ = wrap_model(batch["graph_time"])
            return {"x_t": x_t}
        else:
            raise ValueError("Unsupported derivative order")

    def training_step(self, batch, batch_idx):
        loss_all = 0.0
        loss_clipped_all = 0.0
        results = self.get_derivatives(batch)
        batch_size = batch["reaction_index"].shape[0]
        log_kwargs = dict(
            sync_dist=True, on_step=True, on_epoch=True, batch_size=batch_size
        )
        for loss_name, loss_fn in self.losses.items():
            try:
                loss = loss_fn(batch, results)
            except RuntimeError as e:
                print(f"Error in loss function {loss_name}: {e}")
                if self.trainer.global_rank == 0:
                    torch.save(self.state_dict(), "debug_checkpoint.pt")
                raise e
            loss_clipped = (
                torch.clamp(loss.detach(), max=loss_fn.log_clip_value)
                if loss_fn.log_clip_value is not None
                else loss.detach()
            )
            self.log(f"train/{loss_name}", loss_clipped, **log_kwargs)
            loss_all = loss_all + loss * loss_fn.weight
            loss_clipped_all = loss_clipped_all + loss_clipped * loss_fn.weight
        self.log("train/loss", loss_clipped_all, **log_kwargs)
        return loss_all

    def on_after_backward(self):
        # Skipping updates in case of unstable gradients
        # https://github.com/Lightning-AI/lightning/issues/4956
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                if not valid_gradients:
                    break
        if not valid_gradients:
            if self.trainer.global_rank == 0:
                print("Skipping update due to unstable gradients (NaN or Inf)")
            self.zero_grad()

    def validation_step(self, batch, batch_idx):
        loss_clipped_all = 0.0
        results = self.get_derivatives(batch)
        batch_size = batch["reaction_index"].shape[0]
        log_kwargs = dict(
            sync_dist=True, on_step=False, on_epoch=True, batch_size=batch_size
        )
        for loss_name, loss_fn in self.losses.items():
            loss = loss_fn(batch, results)
            loss_clipped = (
                torch.clamp(loss.detach(), max=loss_fn.log_clip_value)
                if loss_fn.log_clip_value is not None
                else loss.detach()
            )
            self.log(f"val/{loss_name}", loss_clipped, **log_kwargs)
            loss_clipped_all = loss_clipped_all + loss_clipped * loss_fn.weight
        self.log("val/loss", loss_clipped_all, **log_kwargs)
        # NOTE: using the clipped loss for early stopping
        return loss_clipped_all

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
