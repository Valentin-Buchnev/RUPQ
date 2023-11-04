import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from rupq.tools.hooks import DataSaverHook
from rupq.tools.libs import QuantModule


class QuantImagesCallback(pl.callbacks.Callback):
    """
    Callback which adds quantization info to tensorboard at the end of each training epoch:
        - Plots for weights quantization
        - Plots for inputs quantization
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_saver = DataSaverHook(store_input=True)

    def get_weights_quantization_image(self, channel_kernels, channel_quant_bounds, title=None):
        """
        Returns pyplot figure with image of weights quantization.
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        if title:
            plt.title(title, fontsize=24)
        ax.boxplot(channel_kernels)
        ax.fill_between(
            np.arange(1, channel_kernels.shape[1] + 1),
            channel_quant_bounds[0],
            channel_quant_bounds[1],
            color="purple",
            alpha=0.3,
            label="quantization range",
        )
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        plt.legend(fontsize=16)
        plt.xlabel("channels", fontsize=16)
        plt.close(fig)
        return fig

    def get_inputs_quantization_image(self, inputs, quant_bounds, title=None):
        """
        Returns pyplot figure with image of inputs quantization.
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        if title:
            plt.title(title, fontsize=24)
        bins, _, _ = plt.hist(inputs, bins=1000, alpha=0.5, label="inputs")
        ymax = 1.1 * np.sort(bins)[-2]

        ax.fill_between(
            [quant_bounds[0], quant_bounds[1]],
            [ymax, ymax],
            [0, 0],
            color="purple",
            alpha=0.2,
            label="quantization range",
        )
        ax.set_ylim((0, ymax))
        plt.legend(fontsize=16)
        plt.close(fig)
        return fig

    def get_weights_quantization_to_plot(self, module: QuantModule):
        """
        Returns reshaped kernel of shape [C, -1] and quantization bounds
        """
        kernel = torch.clone(module.weight)
        step = module.quantizer.weight_quantizer.get_standardized_step(module.weight)
        channel_size = kernel.shape[0]

        channel_kernels = torch.reshape(kernel, (-1, channel_size))

        if step.numel() != channel_size:
            assert step.numel() == 1, " Expected step of size 1, but got {}".format(step.numel())
            channel_steps = torch.zeros([channel_size])
            channel_steps.fill_(step)
        else:
            channel_steps = step.reshape(-1)

        channel_quant_bounds = (
            (channel_steps.cpu() * module.quantizer.weight_quantizer.negative_clip.item()).detach().numpy(),
            (channel_steps.cpu() * module.quantizer.weight_quantizer.positive_clip.item()).detach().numpy(),
        )
        return channel_kernels.cpu().detach().numpy(), channel_quant_bounds

    def get_inputs_quantization_to_plot(self, module: QuantModule, input: torch.Tensor):
        """
        Returns reshaped input feature map of shape [-1] and quantization bounds.
        """
        if hasattr(module.quantizer.input_quantizer, "offset"):
            input -= module.quantizer.input_quantizer.offset

        step = module.quantizer.input_quantizer.get_standardized_step(input)

        quant_bounds = (
            (step * module.quantizer.input_quantizer.negative_clip).cpu().detach().numpy(),
            (step * module.quantizer.input_quantizer.positive_clip).cpu().detach().numpy(),
        )
        return input.cpu().detach().numpy().reshape(-1), quant_bounds

    def log_quant_images(self, trainer, pl_module):
        """
        Adds weights and inputs quantization plots for all quant layers to tensorboard.
        """
        for name, module in pl_module.model.named_modules():
            if isinstance(module, QuantModule):
                if module.quantizer.weight_quantizer and module.quantizer.weight_quantizer.bit_width < 32:
                    channel_kernels, channel_quant_bounds = self.get_weights_quantization_to_plot(module)
                    weights_quantization_image = self.get_weights_quantization_image(
                        channel_kernels, channel_quant_bounds, title=name
                    )
                    pl_module.logger.experiment.add_figure(
                        "Weights_quantization/{}".format(name),
                        weights_quantization_image,
                        global_step=pl_module.current_epoch,
                        close=True,
                    )

                if module.quantizer.input_quantizer and module.quantizer.input_quantizer.bit_width < 32:
                    handle = module.register_forward_hook(self.input_saver)
                    with torch.no_grad():
                        pl_module.model(pl_module.get_model_inputs(pl_module.logging_batch))

                    module_input = torch.clone(self.input_saver.input[0].squeeze(0))
                    handle.remove()

                    # Keep alive only one process
                    if trainer.global_rank != 0:
                        return

                    inputs, quant_bounds = self.get_inputs_quantization_to_plot(module, module_input)
                    inputs_quantization_image = self.get_inputs_quantization_image(inputs, quant_bounds)
                    pl_module.logger.experiment.add_figure(
                        "Inputs_quantization/{}".format(name),
                        inputs_quantization_image,
                        global_step=pl_module.current_epoch,
                        close=True,
                    )

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch == 0:
            training = pl_module.model.training
            pl_module.model.eval()
            self.log_quant_images(trainer, pl_module)
            if training:
                pl_module.model.train()
