import os
from typing import Optional
import torch

from rupq.dataloaders.coco_dataset import COCODataset
from rupq.dataloaders.datasets_info import DATASETS_INFO
from rupq.dataloaders.transforms.object_detection_augmentations import AUGMENTATION_TRANSFORMS, DEFAULT_TRANSFORMS


class COCODataloader:
    """
    Dataloader for COCO dataset.
    """

    def __init__(
        self, batch_size, image_size: int = 640, multiscale: bool = True, num_workers: Optional[int] = -1, **kwargs
    ):
        """
        Args:
            batch_size (int): Batch size for training loader.
            image_size (int, optional): Input image size. Defaults to 640.
            multiscale (bool, optional): If True, then differrent image sizes are used during training. Defaults to True. 
            num_workers (int, optional): Number of cpu (per each gpu) used for data loading. If -1, then the number of cpus is used. Defaults to -1.
        """

        self.batch_size = batch_size
        self.image_size = image_size
        self.multiscale = multiscale

        self.train_transforms = AUGMENTATION_TRANSFORMS

        self.val_transforms = DEFAULT_TRANSFORMS

        self.num_workers = num_workers if num_workers != -1 else os.cpu_count()
        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        if not self._train_loader:
            dataset = COCODataset(
                DATASETS_INFO["coco"]["train_description_file"],
                img_size=self.image_size,
                multiscale=self.multiscale,
                transform=self.train_transforms,
            )

            self._train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                collate_fn=dataset.collate_fn,
                num_workers=self.num_workers,
            )
        return self._train_loader

    @property
    def val_loader(self) -> torch.utils.data.DataLoader:
        if not self._val_loader:
            dataset = COCODataset(
                DATASETS_INFO["coco"]["val_description_file"],
                img_size=self.image_size,
                multiscale=False,
                transform=self.val_transforms,
            )

            self._val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                collate_fn=dataset.collate_fn,
                num_workers=self.num_workers,
            )
        return self._val_loader
