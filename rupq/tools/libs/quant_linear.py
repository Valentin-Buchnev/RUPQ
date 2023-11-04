import torch
import torch.nn.functional as F

from rupq.tools.libs.quant_module import QuantModule


class QuantLinear(QuantModule):
    """
    Class for quant Linear module.
    """

    def configure(self, module, **kwargs):
        assert type(module) == torch.nn.Linear, "expected module of type torch.nn.Linear, but got {}".format(
            type(module)
        )
        super().configure(module, **kwargs)

    def forward(self, x):
        x, weight = self.quantizer(x, self.weight)

        return F.linear(x, weight, self.bias)
