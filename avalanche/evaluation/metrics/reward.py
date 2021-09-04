from avalanche.evaluation.metric_definitions import GenericPluginMetric, PluginMetric, MetricValue
from avalanche.evaluation.metric_results import MetricResult
from avalanche.evaluation.metrics.mean import WindowedMovingAverage, Mean
from typing import Dict, Union, List
import numpy as np


class MovingWindowedStat(PluginMetric[float]):
    def __init__(self, window_size: int, stat: str = 'mean',
                 name: str = 'Moving Windowed Stat'):
        stat = stat.lower()
        assert stat in ['mean', 'max', 'min', 'std', 'sum']
        self._stat = stat
        self._moving_window = WindowedMovingAverage(window_size)
        self.window_size = window_size
        super().__init__()
        self.x_coord = 0
        self.name = name

    def result(self) -> float:
        if self._stat == 'mean':
            return self._moving_window.result()
        if self._stat == 'max':
            return np.amax(self._moving_window.window, initial=np.float('-inf'))
        if self._stat == 'min':
            return np.amin(self._moving_window.window, initial=np.float('-inf'))
        if self._stat == 'std':
            if len(self._moving_window.window):
                return np.std(self._moving_window.window)
            else:
                return 0.
        if self._stat == 'sum':
            return np.sum(self._moving_window.window)

    def emit(self):
        values = self.result()
        self.x_coord += 1
        return [MetricValue(self, str(self), values, self.x_coord)]

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._moving_window.reset()

    def __str__(self) -> str:
        return f'{self._stat[0].upper()+self._stat[1:].lower()} {self.name} ({self.window_size} steps)'


def moving_window_stat(metric: str, window_size: int = 10, stats=['mean']) -> List[PluginMetric]:
    metric = metric.lower()
    assert metric in ['reward', 'ep_length']
    if metric == 'reward':
        return list(map(lambda s: RewardPluginMetric(window_size, s), stats))
    elif metric == 'ep_length':
        return list(map(lambda s: EpLenghtPluginMetric(window_size, s), stats))


class RewardPluginMetric(MovingWindowedStat):
    """
        Keep track of sum of rewards (returns) per episode.
    """

    def __init__(self, window_size: int, stat: str = 'mean', name: str = 'Reward', *args, **kwargs):
        super().__init__(window_size, stat=stat, name=name, *args, **kwargs)

    def update(self, strategy, is_eval: bool):
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


class EpLenghtPluginMetric(MovingWindowedStat):

    def __init__(self, window_size: int, stat: str = 'mean', name: str = 'Episode Length', *args, **kwargs):
        super().__init__(window_size, stat=stat, name=name, *args, **kwargs)

    def update(self, strategy, is_eval: bool):
        # iterate over parallel envs episodes
        lengths = strategy.eval_ep_lengths if is_eval else strategy.ep_lengths 
        for _, ep_lengths in lengths.items():
            for ep_len in ep_lengths: 
                self._moving_window.update(ep_len)
    # TODO: we could use same system GenericFloatMetric to specify reset callbacks

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


class GenericFloatMetric(PluginMetric[float]):
    # Logs output of a simple float value without many bells and whistles 

    def __init__(self, metric_variable_name: str, name: str,
                 reset_value: float = None, emit_on=['after_rollout'],
                 update_on=['after_rollout']):
        super().__init__()
        self.metric_name = metric_variable_name
        self.name = name
        self.reset_val = reset_value
        self.init_val = None
        self.x_coord = 0
        self.metric_value = None

        # define callbacks
        for update_call in update_on:
            setattr(self, update_call, lambda strat: self._update(strat))
        for emit_call in emit_on:
            foo = getattr(self, emit_call)
            setattr(self, update_call, lambda strat: [
                    f(strat) for f in [foo, lambda s: self._emit(s)]][-1])

    def reset(self) -> None:
        """
        Reset the metric
        """
        if self.reset_val is None:
            self.metric_value = self.init_val
        else:
            self.metric_value = self.reset_val

    def result(self) -> float:
        return self.metric_value

    def _update(self, strategy):
        self.metric_value = getattr(strategy, self.metric_name)
        if self.init_val is None:
            self.init_val = self.metric_value

    def _emit(self, strategy):
        """
        Emit the result
        """
        self.x_coord += 1
        return [MetricValue(self, str(self), self.metric_value, self.x_coord)]

    def __str__(self):
        return self.name
