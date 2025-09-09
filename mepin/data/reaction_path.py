import ase.io
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataLoader

from mepin.tools.geometry import (
    get_neighbor_list_batch,
    kabsch_align,
    random_small_rotation_matrix,
)


class ReactionPathDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/t1x_xtb",
        num_images: int = 8,
        cutoff: float = 6.0,
        split: str = "train",
        seed: int = 42,
        augment_rotation: bool = False,
        augment_angle_scale: float = 0.02,
        frame_alignment: bool = True,
        swap_reactant_product: bool = True,
        deterministic_time: bool = False,
        use_geodesic: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_images = num_images
        self.cutoff = cutoff
        self.rng = np.random.default_rng(seed)
        self.augment_rotation = augment_rotation
        self.augment_angle_scale = augment_angle_scale
        self.frame_alignment = frame_alignment
        self.swap_reactant_product = swap_reactant_product
        self.deterministic_time = deterministic_time
        self.use_geodesic = use_geodesic

        # TODO: currently geodesic paths are generated from aligned structures,
        # so frame_alignment must be True when use_geodesic is True. This is not
        # a strict requirement, so could be fixed in the future.
        if self.use_geodesic:
            if not self.frame_alignment:
                raise ValueError("frame_alignment must be True when use_geodesic")
            if self.swap_reactant_product:
                raise ValueError(
                    "swap_reactant_product must be False when use_geodesic"
                )
            if self.augment_rotation:
                raise ValueError("augment_rotation must be False when use_geodesic")

        # Load the index file
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        self.df = pd.read_csv(f"{data_dir}/split.csv").query(f"split == '{split}'")

        # Load the data
        self.data = []
        for rxn_id in self.df["rxn_id"]:
            reactant = ase.io.read(f"{data_dir}/xyz/{rxn_id}_R.xyz")
            product = ase.io.read(f"{data_dir}/xyz/{rxn_id}_P.xyz")
            if not (
                reactant.get_atomic_numbers() == product.get_atomic_numbers()
            ).all():
                raise ValueError(f"Reactant/product elements do not match for {rxn_id}")

            # NOTE: Kabsch align could behave strangely in float32 precision, so
            # align everything before converting to float32
            reactant_positions = reactant.get_positions()
            product_positions = product.get_positions()
            if self.frame_alignment:
                product_positions = kabsch_align(product_positions, reactant_positions)
            data = {
                "atomic_numbers": reactant.get_atomic_numbers(),
                "reactant_positions": reactant_positions.astype(np.float32),
                "product_positions": product_positions.astype(np.float32),
            }
            if self.use_geodesic:
                # Load the geodesic path
                # NOTE: geodesic interpolation code assumes that the reactant coords
                # are centered, and align image[i + 1] to image[i] for i = 0, 1, ...
                # which result in different behavior from the frame alignment.
                # Therefore, we align every image to the reactant here.
                interp_traj = ase.io.read(f"{data_dir}/geodesic/{rxn_id}.xyz", ":")
                control_positions = [frame.get_positions() for frame in interp_traj]
                data["control_positions"] = kabsch_align(
                    np.array(control_positions), reactant_positions
                ).astype(np.float32)
            self.data.append(data)

    def len(self):
        return len(self.df)

    def get(self, idx):
        # Load the data
        data = self.data[idx]
        atomic_numbers = torch.tensor(data["atomic_numbers"], dtype=torch.long)
        reactant_positions = data["reactant_positions"]
        product_positions = data["product_positions"]

        # Swap the reactant and product positions with 50% probability
        if self.swap_reactant_product and self.rng.random() < 0.5:
            reactant_positions, product_positions = (
                product_positions,
                reactant_positions,
            )

        # Rotational augmentation
        if self.augment_rotation:
            # Rotate the product positions by a random rotation matrix while
            # keeping the center fixed
            rot = random_small_rotation_matrix(self.rng, self.augment_angle_scale)
            center = product_positions.mean(axis=0)
            product_positions = (product_positions - center) @ rot + center

        reactant_positions = torch.from_numpy(reactant_positions)
        product_positions = torch.from_numpy(product_positions)

        # Generate edge indices
        # NOTE: this could be precomputed and stored in the dataset
        edge_index = get_neighbor_list_batch(
            positions_batch=[reactant_positions, product_positions],
            cutoff=self.cutoff,
            lattice=None,
            periodic=False,
        )
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Sample time
        if self.deterministic_time:
            time = np.linspace(0, 1, self.num_images).astype(np.float32)
        else:
            time = self.rng.uniform(0, 1, size=(self.num_images,)).astype(np.float32)
        time = torch.from_numpy(time)

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
        batch["reaction_index"] = torch.tensor([idx], dtype=torch.long)

        if self.use_geodesic:
            batch["control_positions"] = torch.from_numpy(data["control_positions"])

        return batch


class ReactionPathDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/t1x_xtb",
        num_images: int = 8,
        batch_size: int = 1,
        cutoff: float = 6.0,
        seed: int = 42,
        augment_rotation: bool = False,
        augment_angle_scale: float = 0.02,
        frame_alignment: bool = True,
        swap_reactant_product: bool = False,
        use_geodesic: bool = False,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.num_images = num_images
        self.batch_size = batch_size
        self.cutoff = cutoff
        self.seed = seed
        self.augment_rotation = augment_rotation
        self.augment_angle_scale = augment_angle_scale
        self.frame_alignment = frame_alignment
        self.swap_reactant_product = swap_reactant_product
        self.use_geodesic = use_geodesic
        self.num_workers = num_workers

        # TODO: currently geodesic paths are implemented with an assumption that
        # the batch size is 1. Will be generalized in the future.
        if self.use_geodesic and self.batch_size != 1:
            raise ValueError("batch_size must be 1 when use_geodesic is True")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = ReactionPathDataset(
                data_dir=self.data_dir,
                num_images=self.num_images,
                cutoff=self.cutoff,
                split="train",
                seed=self.seed,
                augment_rotation=self.augment_rotation,
                augment_angle_scale=self.augment_angle_scale,
                frame_alignment=self.frame_alignment,
                swap_reactant_product=self.swap_reactant_product,
                deterministic_time=False,  # sample time randomly
                use_geodesic=self.use_geodesic,
            )
            self.val_dataset = ReactionPathDataset(
                data_dir=self.data_dir,
                num_images=self.num_images,
                cutoff=self.cutoff,
                split="val",
                seed=self.seed,
                augment_rotation=False,  # no rotation augmentation for validation
                augment_angle_scale=0.0,
                frame_alignment=self.frame_alignment,
                swap_reactant_product=False,  # no reactant-product swapping
                deterministic_time=True,  # sample time deterministically
                use_geodesic=self.use_geodesic,
            )
        elif stage == "test":
            self.test_dataset = ReactionPathDataset(
                data_dir=self.data_dir,
                num_images=self.num_images,
                cutoff=self.cutoff,
                split="test",
                seed=self.seed,
                augment_rotation=False,  # no rotation augmentation for test
                augment_angle_scale=0.0,
                frame_alignment=self.frame_alignment,
                swap_reactant_product=False,  # no reactant-product swapping
                deterministic_time=True,  # sample time deterministically
                use_geodesic=self.use_geodesic,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
