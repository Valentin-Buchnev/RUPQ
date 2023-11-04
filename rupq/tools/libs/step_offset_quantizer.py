import torch

from rupq.tools.libs.step_quantizer import StepQuantizer


class StepOffsetQuantizer(StepQuantizer):
    """
    Quantizer with trainable step and offset.
    """

    def configure(self, bit_width, half_wave, step_shape=None):
        super().configure(bit_width=bit_width, half_wave=half_wave, step_shape=step_shape)
        self.offset = torch.nn.Parameter(torch.zeros(step_shape or []), requires_grad=True)

    def forward(self, x):
        if self.transformation:
            x, std = self.transformation(x - self.offset)
        else:
            x = x - self.offset

        if not self.initialized:
            self.initialize(x)
            self.initialized.copy_(torch.tensor(True))

        x_quantized = self.grid(x, self.step, self.negative_clip, self.positive_clip)

        if self.transformation:
            x_quantized = self.transformation.inverse(x_quantized, std) + self.offset
        else:
            x_quantized = x_quantized + self.offset

        return x_quantized
