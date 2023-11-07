import os
from typing import Optional
import torch
from torchvision import transforms

from rupq.dataloaders.datasets_info import DATASETS_INFO
from rupq.dataloaders.superresolution_dataset import SuperResolutionDataset
from rupq.dataloaders.transforms.coupled_normalization import CoupledNormalization
from rupq.dataloaders.transforms.coupled_random_rectangular_rotation import CoupledRandomRectangularRotation


class DIV2KDataloader:
    """
    Dataloader for DIV2K dataset.
    """

    def __init__(
        self,
        batch_size: int,
        patch_size: int,
        scale: int,
        normalize: Optional[bool] = True,
        num_workers: Optional[int] = -1,
    ):
        """
        Args:
            batch_size (int): Batch size for training loader.
            patch_size (int): Size of random crop augmentation.
            scale (int): Upscaling factor.
            normalize (bool, optional): Whether to use input data normalization. Defaults to true.
            num_workers (int, optional): Number of cpu (per each gpu) used for data loading. If -1, then the number of cpus is used. Defaults to -1.
        """

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.scale = scale
        self.normalize = normalize
        if self.normalize:
            normalize_transform = CoupledNormalization(mean=127.5, std=51.0)
        else:
            normalize_transform = transforms.Lambda(lambda x: x)

        self.train_transforms = transforms.Compose([CoupledRandomRectangularRotation(), normalize_transform])
        self.val_transforms = normalize_transform

        self.num_workers = num_workers if num_workers != -1 else min(os.cpu_count(), 8)
        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        if not self._train_loader:
            dataset = SuperResolutionDataset(
                DATASETS_INFO["div2k"]["path"],
                num_range=(1, 800),
                scale=self.scale,
                patch_size=self.patch_size,
                transforms=self.train_transforms,
                train=True,
            )

            self._train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
            )
        return self._train_loader

    @property
    def val_loader(self) -> torch.utils.data.DataLoader:
        if not self._val_loader:
            dataset = SuperResolutionDataset(
                DATASETS_INFO["div2k"]["path"],
                num_range=(801, 900),
                scale=self.scale,
                transforms=self.val_transforms,
                train=False,
            )

            self._val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            )
        return self._val_loader
