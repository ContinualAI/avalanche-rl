from avalanche.core import BasePlugin
from avalanche_rl.compat import BaseStrategy


class RLPlugin(BasePlugin):
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
