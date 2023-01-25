from typing import SupportsFloat
from avalanche.evaluation import Metric
from typing import List
import numpy as np

class WindowedMovingAverage(Metric[float]):
    def __init__(self, window_size: int):
        assert window_size > 0, "Window size cannot be negative"
        super().__init__()
        self.window: List[float] = []
        # useful to compute additional stats on window
        self.window_size = window_size

    def update(self, value: SupportsFloat):
        value = float(value)
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window = self.window[1:]

    def result(self) -> float:
        if not len(self.window):
            return np.float("-inf")
        return sum(self.window) / len(self.window)

    def reset(self):
        self.window = []


__all__ = ["WindowedMovingAverage"]
