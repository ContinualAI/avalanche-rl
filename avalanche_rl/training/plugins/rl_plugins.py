from avalanche.core import BasePlugin
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.plugins.evaluation import EvaluationPlugin

class RLStrategyPlugin(BasePlugin):
    """
    Implements RL-specific callbacks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # RLBaseStrategy callbacks
    def before_rollout(self, strategy: 'BaseTemplate', **kwargs):
        pass

    def after_rollout(self, strategy: 'BaseTemplate', **kwargs):
        pass


class RLEvaluationPlugin(EvaluationPlugin):
    """
    An evaluation plugin that adds RL-specific callbacks.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def before_rollout(self, strategy: 'BaseTemplate', **kwargs):
        self._update_metrics(strategy, 'before_rollout')

    def after_rollout(self, strategy: 'BaseTemplate', **kwargs):
        self._update_metrics(strategy, 'after_rollout')