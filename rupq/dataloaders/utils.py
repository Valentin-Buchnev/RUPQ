import torch


def set_channel(image):
    """
    This function Sets third channel equal to 3.
    Need for monochrome images (e.g. in Set14).
    """
    if image.dim() == 2:
        image = torch.unsqueeze(image, dim=2)

    c = image.shape[2]
    if c == 1:
        image = torch.cat([image] * 3, dim=2)

    return image
