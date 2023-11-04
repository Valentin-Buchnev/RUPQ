import pytorch_lightning as pl
import torch
from tqdm import tqdm

from rupq.dataloaders import COCODataloader
from rupq.tools.metrics import MAP


class ValidateCOCOCallback(pl.callbacks.Callback):
    """
    Callback for validating model on COCO.
    """

    def __init__(
        self,
        map: MAP,
        **kwargs,
    ):
        """
        Args:
            map (MAP): class to measure mAP.
            log_every_n_epochs (int, optional): The frequency (in epochs) of validating. Defaults to 5.
        """
        super().__init__(**kwargs)

        self.map = map

    def validate(self, model, pl_module, trainer):
        """
        Validates object detection model and saves the result to logdir (if provided).
        """
        # Keep alive only one process
        if trainer.global_rank != 0:
            return

        device = next(model.parameters()).device
        results = {}

        self.map.reset()
        self.map = self.map.to(device)

        dataset = COCODataloader(batch_size=1, image_size=pl_module.dataset.image_size)
        for batch in tqdm(dataset.val_loader):
            with torch.no_grad():
                self.map.update(model, batch)

            # Keep alive only one process
            if trainer.global_rank != 0:
                return

        results = self.map.compute()

        for key, value in results.items():
            pl_module.logger.experiment.add_scalars(f"val_{key}", {key: value}, global_step=pl_module.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch == trainer.max_epochs - 1:
            training = pl_module.model.training
            pl_module.model.eval()
            self.validate(
                model=pl_module.model.to(pl_module.get_model_inputs(pl_module.logging_batch).device),
                pl_module=pl_module,
                trainer=trainer,
            )
            if training:
                pl_module.model.train()
