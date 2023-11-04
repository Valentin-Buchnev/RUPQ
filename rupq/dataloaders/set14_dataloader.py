import os
from typing import Optional
import torch
from torchvision import transforms

from rupq.dataloaders.datasets_info import DATASETS_INFO
from rupq.dataloaders.superresolution_dataset import SuperResolutionDataset
from rupq.dataloaders.transforms.coupled_normalization import CoupledNormalization


class Set14Dataloader:
    """
    Dataloader for Set14 super-resolution dataset.
    """

    def __init__(self, scale: int, normalize: Optional[bool] = True, num_workers: bool = -1):
        """
        Args:
            scale (int): Upscaling factor.
            normalize (bool, optional): Whether to use input data normalization. Defaults to true.
            num_workers (int, optional): Number of cpu (per each gpu) used for data loading. If -1, then the number of cpus is used. Defaults to -1.
        """

        self.scale = scale
        self.normalize = normalize
        if self.normalize:
            normalize_transform = CoupledNormalization(mean=127.5, std=51.0)
        else:
            normalize_transform = transforms.Lambda(lambda x: x)

        self.transforms = normalize_transform
        self.num_workers = num_workers if num_workers != -1 else os.cpu_count()
        self._dataloader = None

    @property
    def dataloader(self) -> torch.utils.data.DataLoader:
        if not self._dataloader:
            dataset = SuperResolutionDataset(
                path=DATASETS_INFO["set14"]["path"],
                scale=self.scale,
                transforms=self.transforms,
                lr_folder_name="LR_bicubic",
                hr_folder_name="HR",
            )

            self._dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            )
        return self._dataloader
