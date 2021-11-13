from avalanche.evaluation.metric_definitions import PluginMetric, TResult
from avalanche.evaluation.metric_results import MetricResult
from avalanche.training.strategies.base_strategy import BaseStrategy


class RLPluginMetric(PluginMetric[TResult]):
    """
    A plugin metric which adds callbacks for RLBaseStrategy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def before_rollout(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def after_rollout(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass
