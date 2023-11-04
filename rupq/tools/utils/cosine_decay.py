import numpy as np


class CosineDecay:
    """
    Cosine decay with adjustable initial and final values
    """

    def __init__(
        self,
        initial_value: float = 1.0,
        final_value: float = 0.0,
        warmup_steps: int = 0,
        warmup_mode="linear",
        decay_steps: int = 1,
    ):
        """
        Args:
            initial_value (float, optional): Inital value.
            final_value (float, optional): Final value.
            warmup_steps (int, optional): Amount of warmup steps.
            While `step < warmup_step` the returned value is equal to `initial_value`.
            decay_steps (int): Amount of steps after warmup.
            While these steps the returned value gradually changes from `initial_value` to `final_value`.
        """

        self.initial_value = initial_value
        self.final_value = final_value
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode
        self.decay_steps = decay_steps

    def calculate_cosine_value(self, step):
        """
        calculates cosine value which is gradually increasing from 0.0 to 1.0 with step increasing
        """
        relative_step = step / self.decay_steps
        return (1 - np.cos(relative_step * np.pi)) / 2.0

    def __call__(self, step):
        if step < self.warmup_steps:
            if self.warmup_mode == "linear":
                return (step / self.warmup_steps) * self.initial_value
            elif self.warmup_mode == "zero":
                return 0.0
            else:
                raise Exception("{} warmup mode is not implemented!".format(self.warmup_mode))

        cosine_value = self.calculate_cosine_value(step - self.warmup_steps)
        scaled_cosine_value = cosine_value * (self.final_value - self.initial_value) + self.initial_value
        return scaled_cosine_value
