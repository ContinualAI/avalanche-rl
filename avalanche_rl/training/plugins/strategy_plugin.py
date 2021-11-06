from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies import BaseStrategy


class RLStrategyPlugin(StrategyPlugin):
    """
    Implements RL-specific callbacks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # RLBaseStrategy callbacks
    def before_rollout(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_rollout(self, strategy: 'BaseStrategy', **kwargs):
        pass