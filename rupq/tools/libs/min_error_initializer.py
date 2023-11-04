from typing import Optional

import numpy as np
import torch

from rupq.tools.libs.grid import Grid


class MinErrorInitializer(torch.nn.Module):
    """
    Initializer for inputs from LSQ+ paper (https://arxiv.org/abs/2004.09576).
    Works as a brute-force. Calculates Lp reconstruction loss for all steps from uniform grid.
    """

    def __init__(self, p: Optional[float] = 2.0, n_iters: Optional[int] = 300, **kwargs):
        """
        Args:
            p (float, optional): power in Lp loss. Defaults to 2.0.
            n_iters (int, optional): Cardinality of uniform step grid. Defaults to 300.
        """
        super().__init__(**kwargs)
        self.p = p
        self.n_iters = n_iters

    def forward(self, x, negative_clip, positive_clip, step_shape=None):
        step_shape = step_shape or []

        shape = step_shape
        if len(shape) == 0:
            shape = tuple([1] * len(x.shape))

        axes_to_keep, axes_to_reduce = [], []
        size_to_keep = 1
        for i in range(len(shape)):
            if shape[i] > 1:
                axes_to_keep.append(i)
                size_to_keep *= shape[i]
            else:
                axes_to_reduce.append(i)
        x = torch.permute(x, axes_to_keep + axes_to_reduce)
        x = torch.reshape(x, (size_to_keep, -1))

        grid = Grid()

        steps_init = []
        for i in range(x.shape[0]):
            cur_x = x[i]
            step_min = 1e-7
            step_max = (torch.max(torch.abs(cur_x)) / positive_clip).item()
            steps = torch.linspace(step_min, step_max, self.n_iters)

            quant_error = []

            for step in steps:
                quant_x = grid(cur_x, step, negative_clip, positive_clip)
                err = torch.sum(torch.abs(cur_x - quant_x) ** self.p).item()
                quant_error.append(err)
            steps_init.append(steps[np.argmin(quant_error)])
        steps_init = torch.tensor(steps_init)
        return torch.reshape(steps_init, step_shape or []).detach()
