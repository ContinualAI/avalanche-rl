from avalanche.evaluation.metric_definitions import GenericPluginMetric, PluginMetric, MetricValue
from avalanche.evaluation.metric_results import MetricResult
from avalanche.evaluation.metrics.mean import WindowedMovingAverage, Mean
# from avalanche.training.strategies import BaseStrategy circular dep

# FIXME: add max/std/var.. and other stat wrapper


class RewardPluginMetric(PluginMetric[float]):
    def __init__(self, window_size: int):
        if window_size <= 0:
            self._reward = Mean()
        else:
            self._reward = WindowedMovingAverage(window_size)
        self.window_size = window_size
        super().__init__()
        self.x_coord = 0

    def after_rollout(self, strategy) -> None:
        self.update(strategy)

    def after_update(self, strategy) -> 'MetricResult':
        return self.emit()

    def before_eval_exp(self, strategy: 'BaseStrategy') -> 'MetricResult':
        self.reset()

    def after_eval_exp(self, strategy: 'BaseStrategy') -> 'MetricResult':
        self.update(strategy)
        return self.emit()

    def emit(self):
        mean_reward = self.result()
        self.x_coord += 1
        return [MetricValue(self, str(self), mean_reward, self.x_coord)]

    def update(self, strategy):
        for r in strategy.rewards:
            self._reward.update(r)

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._reward.reset()

    def result(self) -> float:
        """
        Emit the result
        """
        return self._reward.result()

    def __str__(self) -> str:
        s = "Avg Reward"
        if self.window_size > 0:
            s += f' ({self.window_size} steps)'
        return s 
