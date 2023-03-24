from collections import defaultdict
from avalanche_rl.evaluation.metric_definitions import RLPluginMetric
from avalanche.evaluation.metric_definitions import MetricValue
from avalanche.evaluation.metric_results import MetricResult
from avalanche_rl.evaluation.metrics.mean import WindowedMovingAverage
from avalanche.training.templates.base import BaseTemplate
from typing import List
import numpy as np


class MovingWindowedStat(RLPluginMetric[float]):
    def __init__(self, window_size: int, stat: str = 'mean',
                 name: str = 'Moving Windowed Stat', mode='train'):
        assert mode in ['train', 'eval']
        stat = stat.lower()
        assert stat in ['mean', 'max', 'min', 'std', 'sum']
        self._stat = stat
        self._moving_window = WindowedMovingAverage(window_size)
        self.window_size = window_size
        super().__init__()
        self.x_coord = 0
        self.name = name
        self._mode = mode

    def result(self) -> float:
        if self._stat == 'mean':
            return self._moving_window.result()
        if self._stat == 'max':
            return np.amax(self._moving_window.window, initial=np.float64('-inf'))
        if self._stat == 'min':
            return np.amin(self._moving_window.window, initial=np.float64('-inf'))
        if self._stat == 'std':
            if len(self._moving_window.window):
                return np.std(self._moving_window.window)
            else:
                return 0.
        if self._stat == 'sum':
            return np.sum(self._moving_window.window)

    def update(self, strategy: 'BaseTemplate'):
        raise NotImplementedError()

    def emit(self):
        # TODO: only emit once we have completed at least one episode 
        # (e.g. we have one return)?
        # you must emit at every timestep or you can't figure out experience
        # lenght when gathering with `get_all_metrics`, that's not great
        values = self.result()
        metric = [MetricValue(self, str(self), values, self.x_coord)]
        self.x_coord += 1
        return metric

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._moving_window.reset()

    def __str__(self) -> str:
        # def camelcase(s: str):
        #     return s[0].upper() + s[1:].lower()
        # return f'[{camelcase(self._mode)}] {camelcase(self._stat)} {self.name} \
        #     (last {self.window_size} steps)'
        return f"{self._mode}/" + \
            "-".join([self.name, self._stat, str(self.window_size)])


def moving_window_stat(
        metric: str, window_size: int = 10, stats=['mean'],
        mode='train') -> List[RLPluginMetric]:
    metric = metric.lower()
    assert metric in ['reward', 'ep_length']
    if mode == 'any':
        mode = ['train', 'eval']
    else:
        mode = [mode]
    metrics = []
    for m in mode:
        if metric == 'reward':
            metrics += list(map(lambda s: ReturnPluginMetric(window_size, s,
                                                             mode=m), stats))
        elif metric == 'ep_length':
            metrics += list(map(lambda s: EpLengthPluginMetric(window_size, s,
                                                               mode=m), stats))
    return metrics

# TODO: immediate reward metric


class ReturnPluginMetric(MovingWindowedStat):
    """
        Keep track of sum of rewards (returns) per episode.
    """
    def __init__(
            self, window_size: int, stat: str = 'mean', name: str = 'reward',
            mode: str = 'train'):
        super().__init__(window_size, stat=stat, name=name, mode=mode)
        self._last_returns_len = 0

    def update(self, strategy):
        rewards = strategy.eval_rewards \
                if self._mode == 'eval' \
                else strategy.rewards 
        # for efficiency, only loop through last *not seen* `window_size
        # returns (full episodes) 
        new_returns = self._moving_window.window_size
        if self._mode == 'train':
            new_returns = len(rewards['past_returns'])-self._last_returns_len
            if new_returns == 0:
                # no episode finished since last update
                return
            new_returns = min(new_returns, self._moving_window.window_size)
            self._last_returns_len = len(rewards['past_returns'])

        for return_ in rewards['past_returns'][-new_returns:]:
            self._moving_window.update(return_)

    # Train
    def before_training_exp(self, strategy: 'BaseTemplate') -> 'MetricResult':
        if self._mode == 'train':
            # reset on new experience
            self.reset()
            self._last_returns_len = 0

    def after_rollout(self, strategy) -> None:
        if self._mode == 'train':
            self.update(strategy)
            return self.emit()

    # Eval
    def before_eval_exp(self, strategy: 'BaseTemplate') -> MetricResult:
        if self._mode == 'eval':
            self.reset()

    def after_eval_exp(self, strategy: 'BaseTemplate') -> MetricResult:
        if self._mode == 'eval':
            self.update(strategy)
            # in eval we always complete at least one episode
            return self.emit()


class EpLengthPluginMetric(MovingWindowedStat):
    def __init__(self, window_size: int, stat: str = 'mean',
                 name: str = 'episodelength', mode='train'):
        super().__init__(window_size, stat=stat, name=name, mode=mode)
        self._actor_ep_lengths = defaultdict(lambda: 0)

    def update(self, strategy):
        lengths = strategy.eval_ep_lengths \
                if self._mode == 'eval' \
                else strategy.ep_lengths 
        new_ep_lengths = self._moving_window.window_size
        # iterate over parallel envs episodes (actor_id->ep_lengths)
        for actor_id, actor_ep_lengths in lengths.items():
            if self._mode == 'train':
                new_ep_lengths = len(
                    actor_ep_lengths)-self._actor_ep_lengths[actor_id]
                if new_ep_lengths == 0:
                    continue
                new_ep_lengths = min(
                    new_ep_lengths, self._moving_window.window_size)
                self._actor_ep_lengths[actor_id] = len(actor_ep_lengths)

            for ep_len in actor_ep_lengths[-new_ep_lengths:]: 
                self._moving_window.update(ep_len)

    # TODO:
    # we could use same system GenericFloatMetricto specify reset callbacks

    # Train
    def before_training_exp(self, strategy: 'BaseTemplate') -> 'MetricResult':
        if self._mode == 'train':
            # reset on new experience
            self.reset()
            self._actor_ep_lengths = defaultdict(lambda: 0)

    def after_rollout(self, strategy) -> None:
        if self._mode == 'train':
            self.update(strategy)
            return self.emit()

    # Eval
    def before_eval_exp(self, strategy: 'BaseTemplate') -> MetricResult:
        if self._mode == 'eval':
            self.reset()

    def after_eval_exp(self, strategy: 'BaseTemplate') -> MetricResult:
        if self._mode == 'eval':
            self.update(strategy)
            return self.emit()


class GenericFloatMetric(RLPluginMetric[float]):
    # Logs output of a simple float value without too many bells and whistles 
    def __init__(self, metric_variable_name: str, name: str,
                 reset_value: float = None, emit_on=['after_rollout'],
                 update_on=['after_rollout'], reset_on=[]):
        super().__init__()
        self.metric_name = metric_variable_name
        self.name = name
        self.reset_val = reset_value
        self.init_val = None
        self.x_coord = 0
        self.metric_value = None

        for update_callback in update_on:
            setattr(self, update_callback, self._update)
        for emit_call in emit_on:
            # append calls defining a pipeline of callbacks
            current_foo = getattr(self, emit_call)
            setattr(self, emit_call, lambda strat: [
                    f(strat) for f in [current_foo, self._emit]][-1])
        for reset_callback in reset_on:
            setattr(self, reset_callback, self.reset)

    def reset(self, strategy) -> None:
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
        import torch
        import numpy as np
        self.metric_value = getattr(strategy, self.metric_name)
        # add support for single item arrays/tensors (e.g. loss)
        if isinstance(
                self.metric_value, torch.Tensor) or isinstance(
                self.metric_value, np.ndarray):
            self.metric_value = self.metric_value.item() 
        if self.init_val is None:
            self.init_val = self.metric_value

    def _emit(self, strategy):
        """
        Emit the result
        """
        metric = [MetricValue(self, str(self), self.metric_value, self.x_coord)]
        self.x_coord += 1
        return metric

    def __str__(self):
        return self.name
