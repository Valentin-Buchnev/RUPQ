import pytorch_lightning as pl

from rupq.tools.hooks import DataSaverHook
from rupq.tools.libs.quant_module import QuantModule


class QuantVarsCallback(pl.callbacks.Callback):
    """
    Callback which adds quantization info to tensorboard at the end of each traning epoch:
        - Quantization steps and offsets
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_saver = DataSaverHook(store_input=True)

    def log_quant_variables(self, trainer, pl_module):
        """
        Adds quantization steps and offsets to tensorboard for all layers.
        """
        for name, module in pl_module.model.named_modules():
            if isinstance(module, QuantModule):
                logs = {}
                if module.quantizer.input_quantizer and hasattr(module.quantizer.input_quantizer, "step"):
                    logs["input_step"] = module.quantizer.input_quantizer.step.mean()
                if module.quantizer.input_quantizer and hasattr(module.quantizer.input_quantizer, "offset"):
                    logs["input_offset"] = module.quantizer.input_quantizer.offset
                if module.quantizer.weight_quantizer and hasattr(module.quantizer.weight_quantizer, "step"):
                    logs["weight_step"] = module.quantizer.weight_quantizer.step.mean()
                pl_module.logger.experiment.add_scalars(
                    "Quant_variables/{}".format(name), logs, global_step=pl_module.current_epoch
                )

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_quant_variables(trainer, pl_module)
