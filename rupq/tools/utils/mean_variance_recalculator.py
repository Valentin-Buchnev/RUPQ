import torch


class MeanVarianceRecalculator:
    """
    Recalculate mean and variance of data given stored mean and variance for previous data and for new batch of data
    """

    def __init__(self, shape: tuple):
        """
        Args:
            shape (tuple): shape for mean and variance
        """
        self.mean = torch.zeros(shape)
        self.variance = torch.zeros(shape)
        self.size = 0.0

    def to(self, device):
        self.mean = self.mean.to(device)
        self.variance = self.variance.to(device)

    def update(self, new_mean, new_variance, new_size):
        prev_second_moment = self.variance + self.mean ** 2
        new_second_moment = new_variance + new_mean ** 2

        size = self.size + new_size
        mean = self.size / size * self.mean + new_size / size * new_mean
        second_moment = self.size / size * prev_second_moment + new_size / size * new_second_moment
        variance = second_moment - mean ** 2

        self.mean = mean
        self.variance = variance
        self.size = size

    def compute(self):
        return self.mean, self.variance * (self.size) / (self.size - 1)
