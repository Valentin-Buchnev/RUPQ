import torch

from rupq.tools.training.training_loop import TrainingLoop


class ObjectDetectionTrainingLoop(TrainingLoop):
    def get_model_inputs(self, batch):
        return batch[1]

    def get_targets(self, batch):
        return batch[2]

    def compute_total_loss(self, model_outputs, batch):

        loss = self.loss(model_outputs, self.get_targets(batch), self.model)
        self.logs["target_loss"] = torch.clone(loss)
        loss *= self.get_model_inputs(batch).shape[0]
        return loss

    def validation_step(self, val_batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass
