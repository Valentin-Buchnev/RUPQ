import torch
import torch.nn.functional as F

from rupq.tools.libs.quant_module import QuantModule


class QuantConv2d(QuantModule):
    """
    Class for quant Conv2d module.
    """

    def configure(self, module, **kwargs):
        assert type(module) in {
            torch.nn.Conv2d,
        }, "expected module of type torch.nn.Conv2d, but got {}".format(type(module))
        super().configure(module, **kwargs)

    def forward(self, x):
        x, weight = self.quantizer(x, self.weight)

        return F.conv2d(
            x,
            weight,
            self.bias,
            self.module.stride,
            self.module.padding,
            self.module.dilation,
            self.module.groups,
        )
