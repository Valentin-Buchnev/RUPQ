import torch


class MeanStdInitializer(torch.nn.Module):
    """
    Initializer for weights from LSQ+ paper (https://arxiv.org/abs/2004.09576).
    """

    def get_mean_per_channel(self, x):
        """
        Returns mean value for each output channel.
        """
        mean_x = torch.mean(x.reshape(x.shape[0], -1), dim=1)
        return mean_x

    def forward(self, x, negative_clip, positive_clip, step_shape=None):
        if step_shape and len(step_shape) > 0:
            mean = x.mean(dim=[i for i in range(1, len(x.shape))], keepdim=True)
            std = x.std(dim=[i for i in range(1, len(x.shape))], keepdim=True)
        else:
            mean = x.mean()
            std = x.std()

        step = torch.max(torch.stack([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]), dim=0)[0]
        step /= positive_clip
        return step.reshape(step_shape or []).detach()
