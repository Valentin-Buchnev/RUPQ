import torch

from rupq.tools.libs.step_quantizer import StepQuantizer


class ModuleQuantizer(torch.nn.Module):
    """
    Quantizer for modules.
    This class quantizes input activations/weights of the module.
    """

    def __init__(self, input_quantizer: StepQuantizer, weight_quantizer: StepQuantizer, **kwargs):
        """
        Args:
            input_quantizer (StepQuantizer): quantizer for input activations.
            weight_quantizer (StepQuantizer): quantizer for weights.
        """
        super().__init__(**kwargs)
        self.input_quantizer = input_quantizer
        self.weight_quantizer = weight_quantizer

    def configure(self, input_bit_width, weight_bit_width, half_wave, step_shape=None):
        self.input_quantizer.configure(bit_width=input_bit_width, half_wave=half_wave)
        self.weight_quantizer.configure(bit_width=weight_bit_width, half_wave=False, step_shape=step_shape)

    def forward(self, inputs, weight):
        inputs = self.input_quantizer(inputs)
        weight = self.weight_quantizer(weight)
        return inputs, weight
