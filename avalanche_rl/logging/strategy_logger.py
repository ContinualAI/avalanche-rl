from avalanche.evaluation.metric_results import MetricValue
from avalanche.logging import BaseLogger
from typing import List
from avalanche_rl.compat import BaseStrategy


class RLStrategyLogger(BaseLogger):
    """
    Strategy logger adding RL-specific callbacks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def before_rollout(self, strategy: "BaseStrategy", metric_values: List["MetricValue"], **kwargs):
        for val in metric_values:
            self.log_metric(val, "before_rollout")

    def after_rollout(self, strategy: "BaseStrategy", metric_values: List["MetricValue"], **kwargs):
        for val in metric_values:
            self.log_metric(val, "after_rollout")


__all__ = ["RLStrategyLogger"]
