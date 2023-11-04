from typing import Optional

import torch


class Standardization(torch.nn.Module):
    """
    Transformation which divides data on its std.
    If momentum > 0.0, then EMA is applied to approximate std.
    It is expected that for the weights momentum = 0.0 and for the input activations momentum > 0.0.
    """

    def __init__(
        self,
        dim: Optional[list] = None,
        momentum: Optional[float] = 0.0,
        std_init: Optional[float] = None,
        trainable: Optional[bool] = True,
        **kwargs,
    ):
        """
        Args:
            dim (list, optional): List of dimentions to reduce while calculating standard deviation.
            If None, then all dimentions are reduced. Defaults to None.
            momentum (float, optional): Momentum for EMA calculation. 0.0 equals to no EMA. Defaults to 0.0.
            std_init (float, optional): Inital value for `ema_std`.
            If None, the std value of first batch is used. Defaults to None.
            trainable: Whether the gradient flows through std. Defaults to True.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.momentum = momentum
        self.std_init = std_init
        self.trainable = trainable

        self.register_buffer("initialized", torch.tensor(False))

    def configure(self, step_shape=None):
        self.register_buffer("ema_std", torch.ones(step_shape or []) * (self.std_init or 1.0))

    def forward(self, x):
        if self.dim:
            std = torch.std(x, self.dim, keepdim=True)
        else:
            std = torch.std(x)

        if not self.trainable:
            std = std.detach()

        if not self.initialized and not self.std_init:
            assert std.shape == self.ema_std.shape, "std initialization's shape must be the same as for std"
            self.ema_std.copy_(std.detach())
            self.initialized.copy_(torch.tensor(True))

        # If this is inference, the EMA std must not be updated
        if not self.training:
            # if momentum is zero, then return std of x
            # if not, return EMA std
            std = self.ema_std.detach() if self.momentum != 0.0 else std.detach()
            return x / std, std

        # Update EMA
        new_std = (self.momentum * self.ema_std).detach() + (1 - self.momentum) * std

        self.ema_std.copy_(new_std.detach())
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.ema_std, op=torch.distributed.ReduceOp.AVG)

        return x / new_std, new_std

    def inverse(self, x, std):
        return x * std
