import os
import torch
import torchvision
from torchvision import transforms

from rupq.dataloaders.datasets_info import DATASETS_INFO


class ImageNet2012Dataloader:
    """
    Dataloader for ImageNet which provides train and validation loader.
    """

    def __init__(self, batch_size: int, num_workers: int = -1):
        """
        Args:
            batch_size (int): Batch size for both training and validation loaders.
            num_workers (int, optional): Number of cpu (per each gpu) used for data loading. If -1, then the number of cpus is used. Defaults to -1.
        """

        self.batch_size = batch_size

        # For normalization, mean and std values are calculated per channel and can be found on the web.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.num_workers = num_workers if num_workers != -1 else os.cpu_count() 
        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        if not self._train_loader:
            dataset = torchvision.datasets.ImageFolder(
                DATASETS_INFO["imagenet2012"]["train_path"],
                transform=self.train_transforms,
            )

            self._train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self._train_loader

    @property
    def val_loader(self) -> torch.utils.data.DataLoader:
        if not self._val_loader:
            dataset = torchvision.datasets.ImageFolder(
                DATASETS_INFO["imagenet2012"]["validation_path"],
                transform=self.val_transforms,
            )

            self._val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self._val_loader
