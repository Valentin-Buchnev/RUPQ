import torch


class Grid(torch.nn.Module):
    """
    LSQ grid (https://arxiv.org/pdf/1902.08153.pdf).
    """

    def grad_scale(self, x, scale):
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad

    def forward(self, x, step, negative_clip, positive_clip):
        step_grad_scale = 1.0 / ((positive_clip * x.numel() / step.numel()) ** 0.5)
        step = self.grad_scale(step, step_grad_scale)
        x = x / step
        x = torch.clamp(x, negative_clip, positive_clip)
        x = (x.round() - x).detach() + x
        x = x * step
        return x
