"""Streaming feature statistics used when preparing a new training corpus."""

import torch


class RunningMoments:
    """Accumulate population mean and standard deviation without retaining data."""

    def __init__(self, dimension: int):
        self.count = 0
        self.sum = torch.zeros(dimension, dtype=torch.float64)
        self.sum_squares = torch.zeros(dimension, dtype=torch.float64)

    def update(self, values: torch.Tensor, valid: torch.Tensor) -> None:
        selected = values.detach()[valid].double().cpu()
        if not len(selected):
            return
        self.merge(len(selected), selected.sum(dim=0), selected.square().sum(dim=0))

    def merge(self, count: int, total: torch.Tensor, total_squares: torch.Tensor) -> None:
        """Merge sufficient statistics computed by another process."""
        if not count:
            return
        self.count += int(count)
        self.sum += total.detach().double().cpu()
        self.sum_squares += total_squares.detach().double().cpu()

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0:
            raise ValueError("Cannot compute statistics without valid frames")
        mean = self.sum / self.count
        variance = (self.sum_squares / self.count - mean.square()).clamp_min(0)
        return mean.float(), variance.sqrt().float()
