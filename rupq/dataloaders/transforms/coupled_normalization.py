from torchvision import transforms


class CoupledNormalization(transforms.Normalize):
    """
    Normalization for image-to-image task.
    Differences from `transforms.Normalize`:
        - Takes raw and target images as input.
    """

    def __call__(self, data):
        raw_image, target_image = data
        return super().forward(raw_image), super().forward(target_image)
