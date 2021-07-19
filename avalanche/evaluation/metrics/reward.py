from avalanche.evaluation.metric_definitions import GenericPluginMetric, PluginMetric, MetricValue
from avalanche.evaluation.metric_results import MetricResult
from avalanche.evaluation.metrics.mean import WindowedMovingAverage, Mean
from typing import Dict, Union, List
import numpy as np


class MovingWindowedStatsPluginMetric(PluginMetric[List[float]]):

    def __init__(
            self, window_size: int, name: str = 'Moving Windowed Stats',
            stats=['mean']):
        assert len(stats) > 0
        self._moving_window = WindowedMovingAverage(window_size)
        self.window_size = window_size
        super().__init__()
        self.x_coord = 0
        self.stats = stats
        self.name = name

    def emit(self):
        values = self.result()
        self.x_coord += 1
        return [MetricValue(self, str(self), values, self.x_coord)]

    def update(self, strategy):
        raise NotImplementedError()

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._moving_window.reset()

    def result(self) -> List[float]:
        """
        Emit the result
        """
        values = []
        for stat in self.stats:
            if 'mean' == stat:
                values.append(self._moving_window.result())
            if 'max' == stat:
                values.append(np.amax(self._moving_window.window, initial=np.float('-inf')))
            if 'min' == stat:
                values.append(np.amin(self._moving_window.window, initial=np.float('-inf')))
            if 'std' == stat:
                values.append(np.std(self._moving_window.window))
        return values

    def __str__(self) -> str:
        s = ""
        for stats in self.stats:
            s += f"{stats[0].upper()+stats[1:]}/"
        s = s[:-1] + f" {self.name}" 
        s += f' ({self.window_size} steps)'
        return s 


class RewardPluginMetric(MovingWindowedStatsPluginMetric):
    # FIXME: this won't ever move from -1 for envs with -1 reward per step
    def __init__(
            self, window_size: int, name: str = 'Reward', eval_ony=False, *args,
            **kwargs):
        super().__init__(window_size, name=name, *args, **kwargs)
        self.eval_only = eval_ony

    def update(self, strategy):
        for r in strategy.rewards:
            self._moving_window.update(r)
    # Train
    def after_rollout(self, strategy) -> None:
        if not self.eval_only:
            self.update(strategy)

    def after_update(self, strategy) -> 'MetricResult':
        if not self.eval_only:
            return self.emit()

    # Eval
    def before_eval_exp(self, strategy: 'BaseStrategy') -> MetricResult:
        self.reset()

    def after_eval_exp(self, strategy: 'BaseStrategy') -> MetricResult:
        self.update(strategy)
        return self.emit()
    
    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        self.reset()


class EpLenghtPluginMetric(MovingWindowedStatsPluginMetric):

    def __init__(self, window_size: int, name: str = 'Episode Length', *args, **kwargs):
        super().__init__(window_size, name=name, *args, **kwargs)

    def update(self, strategy, is_eval:bool):
        # iterate over parallel envs episodes
        lengths = strategy.eval_ep_lengths if is_eval else strategy.ep_lengths 
        for _, ep_lengths in lengths.items():
            for ep_len in ep_lengths: 
                self._moving_window.update(ep_len)
    # Train
    def after_rollout(self, strategy) -> None:
        self.update(strategy, False)

    def after_update(self, strategy) -> 'MetricResult':
        return self.emit()

    # Eval
    def before_eval_exp(self, strategy: 'BaseStrategy') -> MetricResult:
        self.reset()

    def after_eval_exp(self, strategy: 'BaseStrategy') -> MetricResult:
        self.update(strategy, True)
        return self.emit()

    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        self.reset()