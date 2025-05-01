import dxtb
import tad_mctc as mctc
import torch
import torch.nn as nn
from dxtb._src.constants import labels
from torch_geometric.data import Batch


class ConvergenceError(RuntimeError):
    """Error raised when the optimization does not converge."""

    pass


class InvalidPositionsError(RuntimeError):
    """Error raised when the new positions are invalid."""

    pass


class XTBWrapper(nn.Module):
    """Wrapper for the xTB model."""

    def __init__(
        self, params: dxtb.Param, scf_mode: str | int = labels.SCF_MODE_IMPLICIT
    ):
        super().__init__()
        self.params = params
        # NOTE: params are loaded lazily, so we just call name to load them
        # this might not be strictly necessary
        self.name = params.meta.name
        self.scf_mode = scf_mode
        # Disable the timer to avoid timer errors
        dxtb.timer.disable()

    def forward(self, batch: Batch, new_positions: torch.Tensor) -> torch.Tensor:
        """Forward pass through the XTB model."""
        device = batch.batch.device
        dtype = new_positions.dtype
        batch_size = batch["reaction_index"].shape[0]
        num_images = (batch.batch[-1] + 1).item() // batch_size

        # Create input tensors for the dxTB calculator
        input_numbers, input_positions = [], []
        for i in range(batch_size):
            for j in range(num_images):
                idx = i * num_images + j
                node_range = batch.batch == idx
                input_numbers.append(batch.atomic_numbers[node_range])
                input_positions.append(new_positions[node_range])

            # Append the reactant positions for the reference energies
            input_numbers.append(batch.atomic_numbers[node_range])
            input_positions.append(batch.reactant_positions[node_range])

        # Pad the input tensors to the same length
        input_numbers = mctc.batch.pack(input_numbers)
        input_positions = mctc.batch.pack(input_positions) * mctc.units.AA2AU
        input_charges = torch.zeros(
            (input_numbers.shape[0],), dtype=dtype, device=device
        )

        try:
            # Instantiate the calculator and compute the energies
            # NOTE: We use SCF_MODE_IMPLICIT due to memory leakage issues with
            # the default mode, SCF_MODE_IMPLICIT_NON_PURE
            # maybe related: https://github.com/grimme-lab/dxtb/issues/183
            calc = dxtb.Calculator(
                input_numbers,
                self.params,
                opts={"verbosity": 0, "scf_mode": self.scf_mode},
                device=device,
                dtype=dtype,
            )
            energies = calc.get_energy(input_positions, input_charges)

            # Subtract the reference energies
            energies = energies.reshape(batch_size, num_images + 1)
            energies = energies[:, :-1] - energies[:, -1][:, None]
            energies = energies.flatten() * mctc.units.AU2EV

        # Handle errors
        except RuntimeError as e:
            isnan = torch.isnan(input_positions).any()
            del input_numbers, input_positions, input_charges
            msg = str(e)
            if "Fermi energy failed to converge" in msg:
                # This error likely occurs when the positions are valid but not
                # physically meaningful
                raise ConvergenceError("Fermi energy failed to converge.")
            elif "Matrix appears to be not symmetric" in msg:
                # This error likely occurs when the new positions are invalid
                # (e.g., contain NaNs) so the resulting hcore is not symmetric
                if isnan:
                    raise InvalidPositionsError("New positions contain NaNs.")
                # When the error is not due to NaNs, we raise the original error
                # because we've yet to determine the cause
            raise e

        # Clean up the calculator whether or not an error occurred
        finally:
            calc.reset()
            del calc

        return energies
