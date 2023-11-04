import pytorch_lightning as pl
import torch

from rupq.tools.utils import MeanVarianceRecalculator


class BatchNormReestimationCallback(pl.callbacks.Callback):
    """
    Re-estimates BatchNorm statistics using whole training dataset
    """

    def on_validation_epoch_start(self, trainer, pl_module):
        if pl_module.current_epoch != trainer.max_epochs - 1:
            return

        training = pl_module.model.training
        pl_module.model.train()

        org_momentum = {}
        recalculators = {}
        device = next(pl_module.model.parameters()).device
        for name, module in pl_module.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                org_momentum[name] = module.momentum
                module.momentum = 1.0
                recalculators[name] = MeanVarianceRecalculator(module.running_mean.shape)
                recalculators[name].to(device)

        # Run data for estimation
        for x, y in pl_module.dataset.train_loader:
            with torch.no_grad():
                pl_module.model(x.to(device))
            for name, module in pl_module.model.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    recalculators[name].update(module.running_mean, module.running_var, x.shape[0])

        for name, module in pl_module.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                running_mean, running_var = recalculators[name].compute()
                module.running_mean.data.copy_(running_mean)
                module.running_var.data.copy_(running_var)

                # Reset the momentum in case it would be used anywhere else
                module.momentum = org_momentum[name]

        if not training:
            pl_module.model.eval()
