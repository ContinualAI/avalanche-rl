from typing import List
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import stream_type
from avalanche.logging.interactive_logging import InteractiveLogger
from tqdm import tqdm
from avalanche.training import BaseStrategy

from avalanche_rl.logging.strategy_logger import RLStrategyLogger


class TqdmWriteInteractiveLogger(InteractiveLogger, RLStrategyLogger):
    """
    Allows to print out stats to console while updating
    progress bar whitout breaking it.
    """

    def __init__(self, log_every: int = 1):
        super().__init__()
        self.log_every = log_every
        self.step_counter: int = 0

    def print_current_metrics(self):
        sorted_vals = sorted(self.metric_vals.values(),
                             key=lambda x: x[0])
        for name, x, val in sorted_vals:
            val = self._val_to_str(val)
            tqdm.write(f'\t{name} = {val}', file=self.file)

    def before_training_exp(self, strategy: 'BaseStrategy',
                            metric_values: List['MetricValue'], **kwargs):
        super().before_training_exp(strategy, metric_values, **kwargs)
        self._progress.total = strategy.current_experience_steps.value

    def after_training_exp(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'], **kwargs):
        self._end_progress()
        return super().after_training_exp(strategy, metric_values, **kwargs)

    def after_training_iteration(self, strategy: 'BaseStrategy',
                                 metric_values: List['MetricValue'], **kwargs):
        self._progress.update()
        self._progress.refresh()
        super().after_update(strategy, metric_values, **kwargs)
        if self.step_counter % self.log_every == 0:
            self.print_current_metrics()
            self.metric_vals = {}
        self.step_counter += 1

    def before_eval(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'], **kwargs):
        self.metric_vals = {}
        tqdm.write('\n-- >> Start of eval phase << --', file=self.file)

    def before_eval_exp(self, strategy: 'BaseStrategy',
                        metric_values: List['MetricValue'], **kwargs):
        # super().before_eval_exp(strategy, metric_values, **kwargs)
        # self._progress.total = strategy.eval_exp_len
        action_name = 'training' if strategy.is_training else 'eval'
        exp_id = strategy.experience.current_experience
        task_id = strategy.experience.task_label
        stream = stream_type(strategy.experience)
        tqdm.write('-- Starting {} on experience {} (Task {}) from {} stream --'
                   .format(action_name, exp_id, task_id, stream), file=self.file)

    def after_eval_exp(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'after_update')
        self.print_current_metrics()
        exp_id = strategy.experience.current_experience
        tqdm.write(f'> Eval on experience {exp_id} (Task '
                   f'{strategy.experience.task_label}) '
                   f'from {stream_type(strategy.experience)} stream ended.',
                   file=self.file)

    def after_eval(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'], **kwargs):
        tqdm.write('-- >> End of eval phase << --\n', file=self.file)
        # self.print_current_metrics()
        self.metric_vals = {}

    def before_training(self, strategy: 'BaseStrategy',
                        metric_values: List['MetricValue'], **kwargs):
        tqdm.write('-- >> Start of training phase << --', file=self.file)

    def after_training(self, strategy: 'BaseStrategy',
                       metric_values: List['MetricValue'], **kwargs):
        tqdm.write('-- >> End of training phase << --', file=self.file)
