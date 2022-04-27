from avalanche_rl.compat import BaseStrategy
from avalanche.training.plugins.evaluation import EvaluationPlugin


class RLEvaluationPlugin(EvaluationPlugin):
    """
    An evaluation plugin that adds RL-specific callbacks.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def before_rollout(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_rollout')

    def after_rollout(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_rollout')
