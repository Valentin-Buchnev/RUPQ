from typing import Optional, Union

import torch

from rupq.tools.libs import Grid
from rupq.tools.libs.mean_std_initializer import MeanStdInitializer
from rupq.tools.libs.min_error_initializer import MinErrorInitializer
from rupq.tools.libs.standardization import Standardization


class StepQuantizer(torch.nn.Module):
    """
    Quantizer with trainable step.
    """

    def __init__(
        self,
        initializer: Union[MeanStdInitializer, MinErrorInitializer],
        transformation: Optional[Standardization] = None,
        per_channel: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            initializer (MeanStdInitializer or MinErrorInitializer, optional): Initializer for quantization step.
            transformation (BasicTransformation, optional): Transformation for data before quantization.
            per_channel (bool, optional): channel-wise step or shift. Default to False
        """
        super().__init__(**kwargs)

        self.grid = Grid()
        self.initializer = initializer
        self.transformation = transformation
        self.per_channel = per_channel

        self.register_buffer("bit_width", torch.tensor(0))
        self.register_buffer("half_wave", torch.tensor(False))
        self.register_buffer("negative_clip", torch.tensor(0))
        self.register_buffer("positive_clip", torch.tensor(0))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("min_value", torch.tensor(0.0))
        self.register_buffer("max_value", torch.tensor(0.0))

    def configure(self, bit_width, half_wave, step_shape=None):
        if self.transformation:
            self.transformation.configure(step_shape=step_shape)

        self.bit_width.copy_(torch.tensor(bit_width))
        self.half_wave.copy_(torch.tensor(half_wave))
        self.negative_clip.copy_(torch.tensor(0 if half_wave else -(2 ** (bit_width - 1))))
        self.positive_clip.copy_(torch.tensor(2 ** bit_width - 1 if half_wave else 2 ** (bit_width - 1) - 1))
        self.step = torch.nn.Parameter(torch.ones(step_shape or []), requires_grad=True)

    def get_standardized_step(self, x):
        if self.transformation:
            _, std = self.transformation(x)
            return self.step * std
        return self.step

    def initialize(self, x):
        self.min_value.copy_(x.min().detach())
        self.max_value.copy_(x.max().detach())

        step = self.initializer(
            x,
            negative_clip=self.negative_clip,
            positive_clip=self.positive_clip,
            step_shape=self.step.shape,
        )
        assert step.shape == self.step.shape, "Step initialization's shape must be the same as for step"
        self.step.data.copy_(step)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.step, op=torch.distributed.ReduceOp.AVG)

    def forward(self, x):
        if self.transformation:
            x, std = self.transformation(x)

        if not self.initialized:
            self.initialize(x)
            self.initialized.copy_(torch.tensor(True))

        x_quantized = self.grid.forward(x, self.step, self.negative_clip, self.positive_clip)

        if self.transformation:
            x_quantized = self.transformation.inverse(x_quantized, std)

        return x_quantized
