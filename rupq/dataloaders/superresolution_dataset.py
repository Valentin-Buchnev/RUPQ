import glob
import os
import pickle
from typing import Any, Optional

import imageio
import torch
import torch.utils.data as data

from rupq.dataloaders.transforms.sr_random_crop import SRRandomCrop
from rupq.dataloaders.utils import set_channel


class SuperResolutionDataset(data.Dataset):
    """
    Dataset for super-resolution task.
    """

    def __init__(
        self,
        path: str,
        scale: int,
        num_range: Optional[tuple] = None,
        patch_size: Optional[int] = None,
        transforms: Optional[Any] = None,
        train: Optional[bool] = False,
        lr_folder_name: Optional[str] = "DIV2K_train_LR_bicubic",
        hr_folder_name: Optional[str] = "DIV2K_train_HR",
    ):
        """
        Args:
            path (str): Path to dataset.
            scale (int): Upscaling factor.
            num_range (int, optional): range of images ids. If None, then all images from data folder are considered.
            Defaults to None.
            patch_size (int, optional): The size of the patch for HR images.
            For LR images it will be decreased in `scale` times: `patch_size // scale`.
            transforms (any, optional): Pipeline of transformations for images. Defaults to None.
            train (bool, optional): If true, the patching is applied. Defaults to false.
            lr_folder_name (str, optional): Name of the folder with low-resolution images.
            'DIV2K_train_LR_bicubic' for DIV2K dataset and 'LR_bicubic' for Set14 dataset.
            Defaults to 'DIV2K_train_LR_bicubic'.
            hr_folder_name (str, optional): Name of the folder with high-resolution images.
            'DIV2K_train_HR' for DIV2K dataset and 'HR' for Set14 dataset.
            Defaults to 'DIV2K_train_HR'.
        """
        self.num_range = num_range
        self.scale = scale
        self.patch_size = patch_size
        self.transforms = transforms
        self.train = train
        self.lr_folder_name = lr_folder_name
        self.hr_folder_name = hr_folder_name

        self.get_patch = None
        if self.patch_size is not None:
            self.get_patch = SRRandomCrop(patch_size=self.patch_size, scale=self.scale)

        self._set_filesystem(path)

    def _copy_to_bin(self, images_list):
        """
        Copies .png images to binary format.
        """
        bin_images_list = []
        for filename in images_list:
            bin_filename = filename.replace(self.path, self.path_bin)
            bin_filename = bin_filename.replace(".png", ".pt")
            self._check_and_load(bin_filename, filename)
            bin_images_list.append(bin_filename)

        return bin_images_list

    def _is_num_in_range(self, path):
        if not self.num_range:
            return True

        # extracts the number 111 from 'path/to/dataset/Xs/0111*.png'
        num = int(path.split("/")[-1][:4])

        return num >= self.num_range[0] and num <= self.num_range[1]

    def _set_filesystem(self, path):
        """
        Store paths to binaries, low and high resolution images.
        """
        self.path = path
        self.path_bin = os.path.join(self.path, "bin")
        self.path_low_resolution = os.path.join(self.path, self.lr_folder_name, "X{}".format(self.scale))
        self.path_high_resolution = os.path.join(self.path, self.hr_folder_name)

        os.makedirs(self.path_bin, exist_ok=True)
        os.makedirs(os.path.join(self.path_low_resolution.replace(self.path, self.path_bin)), exist_ok=True)
        os.makedirs(self.path_high_resolution.replace(self.path, self.path_bin), exist_ok=True)

        self.images_low_resolution = self._copy_to_bin(
            sorted([x for x in glob.glob(os.path.join(self.path_low_resolution, "*.png")) if self._is_num_in_range(x)])
        )
        self.images_high_resolution = self._copy_to_bin(
            sorted(
                [x for x in glob.glob(os.path.join(self.path_high_resolution, "*.png")) if self._is_num_in_range(x)]
            )
        )

    def _check_and_load(self, bin_filename, filename):
        """
        Checks that binary file exists. If not, creates one.
        """
        if not os.path.isfile(bin_filename):
            print("Making a binary: {}".format(bin_filename))
            with open(bin_filename, "wb") as f:
                pickle.dump(imageio.imread(filename), f)

    def __len__(self):
        return len(self.images_high_resolution)

    def _load_file(self, idx):
        """
        Loads low-resolution and high-resolution images.
        """
        file_low_resolution = self.images_low_resolution[idx]
        file_high_resolution = self.images_high_resolution[idx]

        with open(file_low_resolution, "rb") as f:
            low_resolution = pickle.load(f)
        with open(file_high_resolution, "rb") as f:
            high_resolution = pickle.load(f)

        # Crop HR image, need for some set14 images
        h, w = low_resolution.shape[:2]
        high_resolution = high_resolution[0 : h * self.scale, 0 : w * self.scale]

        return low_resolution, high_resolution

    def __getitem__(self, idx):
        """
        Returns low and high resolution images.
        """
        low_resolution, high_resolution = self._load_file(idx)

        if self.train:
            low_resolution, high_resolution = self.get_patch(low_resolution, high_resolution)

        # Cast to torch.Tensor
        low_resolution = torch.from_numpy(low_resolution).float()
        high_resolution = torch.from_numpy(high_resolution).float()

        # Here the images have shape [H, W, C] or [H, W] (for some images from Set14 for example).
        # Lets bring them to the shape [H, W, C].
        low_resolution = set_channel(low_resolution)
        high_resolution = set_channel(high_resolution)

        # Reshape to torch format [C, H, W]
        low_resolution = torch.permute(low_resolution, (2, 0, 1))
        high_resolution = torch.permute(high_resolution, (2, 0, 1))

        # Here the images have desired shape [H, W, 3]
        if self.transforms:
            low_resolution, high_resolution = self.transforms((low_resolution, high_resolution))

        return low_resolution, high_resolution
