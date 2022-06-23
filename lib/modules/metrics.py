import torchmetrics
from typing import *
from torch import Tensor


class RescaledMAE(torchmetrics.MeanAbsoluteError):
    def __init__(self, compute_on_step: Optional[bool] = None, rescale: float = 1):
        super().__init__(compute_on_step=compute_on_step)
        self.rescale = rescale

    def compute(self) -> Tensor:
        return super().compute() * self.rescale
