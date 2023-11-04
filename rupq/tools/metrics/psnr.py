import torch
import torchmetrics
from torchmetrics import functional as F


def convert_to_default_image_range(image, min_val, max_val):
    return torch.clamp((image - min_val) / (max_val - min_val), 0.0, 1.0) * 255.0


def remove_boundary(image, boundary_size):
    return image[..., boundary_size:-boundary_size, boundary_size:-boundary_size]


def get_luminance(image):
    luma_scales = [65.481, 128.553, 24.966]
    luma_scales = image.new_tensor(luma_scales).view(1, 3, 1, 1)
    return (image.mul(luma_scales) / 255.0 + 16.0).sum(dim=1)


class PSNR(torchmetrics.Metric):
    def __init__(self, min_val=-2.5, max_val=2.5, boundary_size=4, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.boundary_size = boundary_size

        self.psnr_sum = 0.0
        self.count = 0

    @property
    def full_state_update(self):
        return False

    def reset(self):
        self.psnr_sum = 0.0
        self.count = 0

    def update(self, y_pred, y_true):
        y_true = convert_to_default_image_range(
            remove_boundary(y_true, self.boundary_size), self.min_val, self.max_val
        )
        y_pred = convert_to_default_image_range(
            remove_boundary(y_pred, self.boundary_size), self.min_val, self.max_val
        )

        y_true = get_luminance(y_true).unsqueeze(1)
        y_pred = get_luminance(y_pred).unsqueeze(1)

        self.psnr_sum += F.peak_signal_noise_ratio(y_pred, y_true, data_range=255.0)
        self.count += 1

    def compute(self):
        return self.psnr_sum / self.count
