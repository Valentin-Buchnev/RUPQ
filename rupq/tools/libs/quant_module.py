from abc import ABCMeta

import torch

from rupq.tools.libs.module_quantizer import ModuleQuantizer


class QuantModule(torch.nn.Module, metaclass=ABCMeta):
    """
    Class for quant modules.
    """

    def __init__(
        self,
        quantizer: ModuleQuantizer,
        **kwargs,
    ):
        """
        Args:
            quantizer (ModuleQuantizer): Module quantizer (conv2d or linear).
        """
        super().__init__(**kwargs)

        self.quantizer = quantizer

    def get_step_shape(self, x, dims):
        if dims is None:
            return []
        return list(x.shape[i] if i in dims else 1 for i in range(x.dim()))

    def configure(self, module, input_bit_width, weight_bit_width, half_wave):
        assert hasattr(module, "weight"), "module {} is assumed to have `weight` attribute".format(type(module))
        assert hasattr(module, "bias"), "module {} is assumed to have `bias` attribute".format(type(module))

        self.module = module
        self.quantizer.configure(
            input_bit_width=input_bit_width,
            weight_bit_width=weight_bit_width,
            half_wave=half_wave,
            step_shape=self.get_step_shape(
                module.weight, dims=[0] if self.quantizer.weight_quantizer.per_channel else None
            ),
        )

    @property
    def weight(self):
        return self.module.weight

    @property
    def bias(self):
        return self.module.bias
