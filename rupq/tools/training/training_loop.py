from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from prettytable import PrettyTable

from rupq.tools.libs import QuantModule


class TrainingLoop(pl.LightningModule):
    """
    Class containing training pipeline which is appropriate for both float and quantization-aware training.
    """

    def __init__(
        self,
        dataset,
        loss: Optional[Any] = None,
        train_metrics: Optional[Dict] = None,
        val_metrics: Optional[Dict] = None,
        print_calibration: Optional[bool] = True,
        **kwargs,
    ):
        """
        Args:
            dataset: Dataset for training and evaluation.
            loss (optional): Target loss.
            train_metrics (dict, optional): Dict of metrics for train. Defaults to None.
            val_metrics (dict, optional): Dict of metrics for validation. Defaults to None.
            print_calibration(bool, optional): If True, prints calibration report at the beginning of the training.
            Defaults to True.
        """
        super().__init__(**kwargs)
        self.dataset = dataset

        self.loss = loss
        train_metrics = train_metrics or {}
        val_metrics = val_metrics or {}
        self.train_metrics_names = list(train_metrics.keys())
        self.val_metrics_names = list(val_metrics.keys())
        for name, metric in train_metrics.items():
            setattr(self, "train_" + name, metric)
        for name, metric in val_metrics.items():
            setattr(self, "val_" + name, metric)

        self.logging_batch = None
        self.print_calibration = print_calibration

    @property
    def automatic_optimization(self):
        return False

    def set_model(self, model):
        self.model = model

    def set_optimizers(self, optimizers, schedulers):
        self.optimizer = optimizers
        self.scheduler = schedulers

    def configure_optimizers(self):
        return self.optimizer, self.scheduler

    def log_losses(self, prefix="", progress_bar=True):
        for name, value in self.logs.items():
            self.log(prefix + name, value, on_step=False, on_epoch=True, sync_dist=True)
            if progress_bar:
                self.log("pb_" + prefix + name, value, logger=False, on_step=True, prog_bar=True, sync_dist=True)

    def log_metrics(self, predictions, targets, prefix="", progress_bar=True):
        for name in getattr(self, prefix + "metrics_names"):
            getattr(self, prefix + name).update(predictions, targets)

        if progress_bar:
            for name in getattr(self, prefix + "metrics_names"):
                self.log(
                    "pb_" + prefix + name,
                    getattr(self, prefix + name).compute(),
                    logger=False,
                    on_step=True,
                    prog_bar=True,
                    sync_dist=True,
                )

    def print_calibration_report(self):
        if not self.print_calibration or self.global_rank != 0:
            return
        table = PrettyTable(["Module", "Tensor", "Bit width", "Half wave", "Min", "Max"])
        table.print_empty = False
        table.float_format = ".4"

        for name, module in self.model.named_modules():
            if isinstance(module, QuantModule):
                input_quantizer_row = [
                    name,
                    "input activation",
                    module.quantizer.input_quantizer.bit_width.item(),
                    module.quantizer.input_quantizer.half_wave.item(),
                    module.quantizer.input_quantizer.min_value.item(),
                    module.quantizer.input_quantizer.max_value.item(),
                ]
                table.add_row(input_quantizer_row)
                weight_quantizer_row = [
                    name,
                    "weight",
                    module.quantizer.weight_quantizer.bit_width.item(),
                    module.quantizer.weight_quantizer.half_wave.item(),
                    module.quantizer.weight_quantizer.min_value.item(),
                    module.quantizer.weight_quantizer.max_value.item(),
                ]
                table.add_row(weight_quantizer_row)
        print(table)

    def compute_total_loss(self, model_outputs, batch):
        loss = self.loss(model_outputs, self.get_targets(batch))

        self.logs["target_loss"] = torch.clone(loss)
        return loss

    def optimization_step(self, loss):
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        for optimizer in optimizers:
            optimizer.zero_grad()
        self.manual_backward(loss)

        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()

    def get_model_inputs(self, batch):
        return batch[0]

    def get_targets(self, batch):
        return batch[1]

    def training_step(self, train_batch, batch_idx, logs=None):
        self.logs = {}
        model_outputs = self.model(self.get_model_inputs(train_batch))

        loss = self.compute_total_loss(model_outputs, train_batch)

        if self.logging_batch is None:
            self.print_calibration_report()
            self.logging_batch = train_batch

        self.optimization_step(loss)
        self.log_losses(prefix="train_")
        self.log_metrics(model_outputs, self.get_targets(train_batch), prefix="train_")

        return loss

    def training_epoch_end(self, outs):
        for name in self.train_metrics_names:
            self.log("train_" + name, getattr(self, "train_" + name).compute(), sync_dist=True)
            getattr(self, "train_" + name).reset()

    def validation_step(self, val_batch, batch_idx):
        self.logs = {}
        model_outputs = self.model(self.get_model_inputs(val_batch))

        loss = self.compute_total_loss(model_outputs, val_batch)
        self.log_losses(prefix="val_", progress_bar=False)
        self.log_metrics(model_outputs, self.get_targets(val_batch), prefix="val_", progress_bar=False)

        return loss

    def validation_epoch_end(self, outputs):
        for name in self.val_metrics_names:
            self.log("val_" + name, getattr(self, "val_" + name).compute(), sync_dist=True)
            getattr(self, "val_" + name).reset()
