"""
The :py:mod:`training` module provides a generic continual learning training
class (:py:class:`BaseStrategy`) and implementations of the most common
CL strategies. These are provided either as standalone strategies in
:py:mod:`training.strategies` or as plugins (:py:mod:`training.plugins`) that
can be easily combined with your own strategy.
"""
from avalanche_rl.evaluation.metrics.reward import moving_window_stat
from avalanche_rl.logging.interactive_logging import TqdmWriteInteractiveLogger
from avalanche_rl.training.plugins.evaluation import RLEvaluationPlugin

default_rl_logger = RLEvaluationPlugin(
                                     moving_window_stat('reward', window_size=10, stats=['mean', 'max', 'std']),
                                     moving_window_stat('reward', window_size=4, stats=['mean', 'std'], mode='eval'),
                                     moving_window_stat('ep_length', window_size=10, stats=['mean', 'max', 'std']),
                                     moving_window_stat('ep_length', window_size=4, stats=['mean', 'std'], mode='eval'),
                                     loggers=[TqdmWriteInteractiveLogger(log_every=10)])
