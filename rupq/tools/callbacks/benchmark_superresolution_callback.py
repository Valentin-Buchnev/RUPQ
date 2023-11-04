from typing import Optional

import pytorch_lightning as pl
import torch

from rupq.dataloaders import Set14Dataloader
from rupq.tools.metrics import PSNR


class BenchmarkSuperresolutionCallback(pl.callbacks.Callback):
    """
    Callback for benchmarking super-resolution model at the end of the training.
    """

    def __init__(
        self,
        psnr: PSNR,
        scale: int,
        normalize: Optional[bool] = True,
        **kwargs,
    ):
        """
        Args:
            psnr (PSNR): class to measure psnr.
            scale (int): Upscale factor.
            normalize (bool, optional): Whether to use input data normalization. Defaults to true.
            log_every_n_epochs (int, optional): The frequency (in epochs) of benchmarking. Defaults to 50.
        """
        super().__init__(**kwargs)

        self.psnr = psnr
        self.benchmark_dataloader = Set14Dataloader(scale=scale, normalize=normalize)

    def benchmark(self, model, pl_module, trainer):
        """
        Benchmarks super-resolution model and saves the result to logdir (if provided).
        """
        # Keep alive only one process
        if trainer.global_rank != 0:
            return

        device = next(model.parameters()).device

        self.psnr.reset()
        self.psnr = self.psnr.to(device)

        for x, y in self.benchmark_dataloader.dataloader:
            with torch.no_grad():
                self.psnr.update(model(x.to(device)), y.to(device))

            # Keep alive only one process
            if trainer.global_rank != 0:
                return

        set14_psnr = self.psnr.compute().item()

        print("Set14 psnr: ", set14_psnr)
        pl_module.logger.experiment.add_scalars(
            "set14_PSNR", {"PSNR": set14_psnr}, global_step=pl_module.current_epoch
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch == trainer.max_epochs - 1:
            training = pl_module.model.training
            pl_module.model.eval()
            self.benchmark(
                model=pl_module.model.to(pl_module.get_model_inputs(pl_module.logging_batch).device),
                pl_module=pl_module,
                trainer=trainer,
            )
            if training:
                pl_module.model.train()
