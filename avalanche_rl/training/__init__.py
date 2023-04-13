"""
The :py:mod:`training` module provides a generic continual learning training
class (:py:class:`BaseStrategy`) and implementations of the most common
CL strategies. These are provided either as standalone strategies in
:py:mod:`training.strategies` or as plugins (:py:mod:`training.plugins`) that
can be easily combined with your own strategy.
"""
from avalanche_rl.evaluation.metrics.reward import moving_window_stat
from avalanche_rl.logging.interactive_logging import TqdmWriteInteractiveLogger
from avalanche_rl.training.plugins.rl_plugins import RLEvaluationPlugin
from typing import List

default_rl_logger = RLEvaluationPlugin(
                                     moving_window_stat('reward', window_size=10, stats=[
                                                        'mean', 'max', 'std']),
                                     moving_window_stat('reward', window_size=4, stats=[
                                                        'mean', 'std'], mode='eval'),
                                     moving_window_stat('ep_length', window_size=10, stats=[
                                                        'mean', 'max', 'std']),
                                     moving_window_stat('ep_length', window_size=4, stats=[
                                                        'mean', 'std'], mode='eval'),
                                     loggers=[TqdmWriteInteractiveLogger(log_every=10)])


def make_logger(log_every: int, train_window_size: int = 10, eval_window_size: int = 4, tracked_metrics: List[str] = ['reward', 'ep_length'], tracked_stats: List[str] = ['mean', 'max', 'std']) -> RLEvaluationPlugin:
    """Creates a metric logger which can be used as `evaluator` to track, record and print out metrics.

    Args:
        log_every (int): How often should we print out metrics, in iteration steps.
        train_window_size (int, optional): Size of the `moving_window_stat` metric tracker during training. Defaults to 10.
        eval_window_size (int, optional): Size of the `moving_window_stat` metric tracker during evaluation. Defaults to 4.
        tracked_metrics (List[str], optional): Metrics to track. Defaults to ['reward', 'ep_length'].
        tracked_stats (List[str], optional): Aggregated stats to compute on the moving window. Defaults to ['mean', 'max', 'std'].

    Returns:
        RLEvaluationPlugin: _description_
    """
    # TODO: add tensorboard flag and generic float tracker 
    metrics = []
    for m in tracked_metrics:
        if train_window_size is not None:
            metrics.append(moving_window_stat(
                m, train_window_size, tracked_stats, mode='train'))
        if eval_window_size is not None:
            metrics.append(moving_window_stat(
                m, eval_window_size, tracked_stats, mode='eval'))

    return RLEvaluationPlugin(*metrics, loggers=[TqdmWriteInteractiveLogger(log_every=log_every)])
