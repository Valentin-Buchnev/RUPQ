from typing import Optional

import pytorch_lightning as pl
import torch

from rupq.models import create_model
from rupq.tools.callbacks import ModelSaverCallback, QuantVarsCallback
from rupq.tools.model_parsing.model_parser import ModelParser
from rupq.tools.utils import get_params


class Task:
    """
    This is a task for model training.
    """

    def __init__(
        self,
        training_loop,
        epochs,
        model_saver,
        model_arch: Optional[str] = None,
        optimizers: Optional[dict] = None,
        model_parser: Optional[ModelParser] = None,
        check_val_every_n_epochs: Optional[int] = 1,
        callbacks: Optional[list] = None,
    ):
        """
        Args:
            training_loop: PL module which will be provided to PL trainer.
            epochs: Number of epochs to train.
            model_saver: Kwargs for saving checkpoint.
            model_arch (str, optional): Architecture of model. Defaults to None.
            optimizers (dict, optional): Dict of optimizers. Defaults to None.
            model_parser (ModelParser, optional): Parser for model quantization. Defaults to None.
            check_val_every_n_epochs (int, optional): The frequency (in epochs) of validation evaluating
            during training. Defaults to None.
            callbacks (list, optional): List of callbacks for PL training. Defaults to None.
        """
        self.training_loop = training_loop
        self.epochs = epochs
        self.model_saver = model_saver

        self.model_arch = model_arch

        self.optimizers = optimizers
        self.model_parser = model_parser
        self.check_val_every_n_epochs = check_val_every_n_epochs
        self.callbacks = callbacks or []
        self.callbacks.append(QuantVarsCallback())

    def set_model(self, load_model_path):
        if load_model_path:
            if self.model_arch:
                raise RuntimeError(
                    "Please, do not provide the model arch in the task constructor "
                    "(or remove the model arch definition from a yaml configuration file) "
                    "because you are loading a pre-trained model."
                )
            model = torch.load(load_model_path)
        else:
            if not self.model_arch:
                raise RuntimeError(
                    "Please, provide the model arch in the task constructor (or in a yaml configuration file)."
                )
            model = create_model(self.model_arch)

        if self.model_parser:
            model = self.model_parser.parse(model)
        self.training_loop.set_model(model)

    def set_optimizers(self):
        optimizers = []
        schedulers = []
        for _, opt in self.optimizers.items():
            optimizer_class = getattr(torch.optim, opt["name"])
            optimizer = optimizer_class(
                params=get_params(self.training_loop.model, params_type=opt.get("model_params_type")), **opt["params"]
            )

            scheduler_class = getattr(torch.optim.lr_scheduler, opt["scheduler"]["name"])
            scheduler = scheduler_class(optimizer, **opt["scheduler"]["params"])

            optimizers.append(optimizer)
            schedulers.append(scheduler)

        self.training_loop.set_optimizers(optimizers=optimizers, schedulers=schedulers)

    def configure(self, logdir=None, load_model_path=None, num_gpus=1, checkpoint_path=None):

        self.set_model(load_model_path)
        self.set_optimizers()

        # The batch size comes from config where we expect the effective batch size,
        # whereas pytorch lightning expects the batch size per gpu.
        self.training_loop.dataset.batch_size //= num_gpus

        # logger
        logger = pl.loggers.TensorBoardLogger(logdir, name=None)

        # callbacks
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        model_saver = ModelSaverCallback(logdir, **self.model_saver)
        progress_bar = pl.callbacks.RichProgressBar()

        # pl.Trainer
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=num_gpus,
            max_epochs=self.epochs,
            logger=logger,
            callbacks=[lr_monitor, model_saver, progress_bar] + self.callbacks,
            check_val_every_n_epoch=self.check_val_every_n_epochs,
        )

        self.checkpoint_path = checkpoint_path

    def run(self):
        self.trainer.fit(
            self.training_loop,
            self.training_loop.dataset.train_loader,
            self.training_loop.dataset.val_loader,
            ckpt_path=self.checkpoint_path,
        )
