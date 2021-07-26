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
                if len(self._moving_window.window):
                    values.append(np.std(self._moving_window.window))
                else:
                    values.append(0.)
        return values

    def __str__(self) -> str:
        s = ""
        for stats in self.stats:
            s += f"{stats[0].upper()+stats[1:]}/"
        s = s[:-1] + f" {self.name}" 
        s += f' ({self.window_size} steps)'
        return s 


class RewardPluginMetric(MovingWindowedStatsPluginMetric):
    """
        Keep track of sum of rewards (returns) per episode.
    """

    def __init__(
            self, window_size: int, name: str = 'Reward', *args,
            **kwargs):
        super().__init__(window_size, name=name, *args, **kwargs)

    def update(self, strategy, is_eval:bool):
        rewards = strategy.eval_rewards if is_eval else strategy.rewards 
        for return_ in rewards['past_returns']:
            self._moving_window.update(return_)

    def before_training_exp(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # reset on new experience
        self.reset()

    # Train
    def after_rollout(self, strategy) -> None:
        self.update(strategy, False)
        return self.emit()

    # Eval
    def before_eval_exp(self, strategy: 'BaseStrategy') -> MetricResult:
        self.reset()

    def after_eval_exp(self, strategy: 'BaseStrategy') -> MetricResult:
        self.update(strategy, True)
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

    def before_training_exp(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # reset on new experience
        self.reset()
        
    # Train
    def after_rollout(self, strategy) -> None:
        self.update(strategy, False)
        return self.emit()

    # Eval
    def before_eval_exp(self, strategy: 'BaseStrategy') -> MetricResult:
        self.reset()

    def after_eval_exp(self, strategy: 'BaseStrategy') -> MetricResult:
        self.update(strategy, True)
        return self.emit()

    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        self.reset()

# TODO: make general for any kind of float metric
class ExplorationEpsilon(PluginMetric[float]):

    def __init__(self):
        super().__init__()

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()
        # current x values for the metric curve
        self.x_coord = 0
        self.eps = 1.

    def reset(self) -> None:
        """
        Reset the metric
        """
        self.eps = 1.

    def result(self)->float:
        return self.eps

    def after_rollout(self, strategy) -> 'MetricResult':
        self.eps = strategy.eps
        return self.emit(strategy)

    def emit(self, strategy):
        """
        Emit the result
        """
        value = self.eps
        self.x_coord += 1 # increment x value
        return [MetricValue(self, str(self), value,self.x_coord)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        return "Exploration Eps"