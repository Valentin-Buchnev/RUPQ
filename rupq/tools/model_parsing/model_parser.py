from copy import deepcopy
from typing import Dict

from rupq.logger import get_logger

logger = get_logger()


class ModelParser:
    """
    Model parser, which replaces FP32 modules to quantized modules, instances of QuantModule.
    """

    def __init__(
        self,
        wrappers: Dict,
        input_bit_width: Dict,
        weight_bit_width: Dict,
        half_wave: Dict,
    ):
        """
        Args:
            wrappers (dict): Dict where torch.nn module is a key, BasicQuantLayer is a value.
            Only `Conv2d` or `Linear` are supported.
            input_bit_width (dict): Dict with info about bit widths for inputs. Has the following structure:
                default (int): default value.
                exceptions (dict):
                    layer_name: value.
            weight_bit_width (dict): Dict with info about bit widths for weights.
            Has the same structure as `input_bit_width`.
            half_wave (dict): Dict with info about half wave. Has the same structure as `input_bit_width`.
        """
        self.wrappers = wrappers
        self.input_bit_width = input_bit_width
        self.weight_bit_width = weight_bit_width
        self.half_wave = half_wave

    def get_value(self, attr_name, submodule_prefix):
        try:
            return getattr(self, attr_name)["exceptions"][submodule_prefix]
        except KeyError:
            assert "default" in getattr(self, attr_name), "Please, provide default option for {}".format(attr_name)
            return getattr(self, attr_name)["default"]

    def modify_module(self, module_name, module):
        """
        The matched quant layers generation
        """
        input_bit_width = self.get_value("input_bit_width", module_name)
        weight_bit_width = self.get_value("weight_bit_width", module_name)
        half_wave = self.get_value("half_wave", module_name)

        assert input_bit_width >= 2 and input_bit_width < 32, "Unsupported input bitwidth: {}".format(input_bit_width)
        assert weight_bit_width >= 2 and weight_bit_width < 32, "Unsupported weight bitwidth: {}".format(
            weight_bit_width
        )
        if module_name in self.wrappers:
            quant_module = deepcopy(self.wrappers[module_name])
        else:
            quant_module = deepcopy(self.wrappers[module._get_name()])
        quant_module.configure(
            module=module, input_bit_width=input_bit_width, weight_bit_width=weight_bit_width, half_wave=half_wave
        )
        logger.info("%s module is %s-%s quantized.", module_name, input_bit_width, weight_bit_width)

        return quant_module

    def is_modifiable_module(self, module, module_name):
        if module_name in self.wrappers:
            return True
        module_type = module._get_name()
        if module_type in self.wrappers:
            return True
        return False

    def modify(self, input_model, modified_modules=None, prefix=""):
        """
        Break down the model (or another module) into modules and change modifiable modules recursively.
        """
        if modified_modules is None:
            modified_modules = set()

        if input_model not in modified_modules:
            modified_modules.add(input_model)
            for name, module in input_model._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name

                if self.is_modifiable_module(module, submodule_prefix):
                    modified_module = self.modify_module(submodule_prefix, module)
                    input_model._modules[name] = modified_module

                self.modify(module, modified_modules=modified_modules, prefix=submodule_prefix)
        return input_model

    def parse(self, input_model):
        return self.modify(input_model)
