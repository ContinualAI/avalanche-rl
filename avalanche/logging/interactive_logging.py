################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 2020-01-25                                                             #
# Author(s): Antonio Carta, Lorenzo Pellegrini                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
import sys
from typing import List, TYPE_CHECKING

from avalanche.evaluation.metric_results import MetricValue
from avalanche.logging import TextLogger
from avalanche.evaluation.metric_utils import stream_type

from tqdm import tqdm

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy


class InteractiveLogger(TextLogger):
    """
    The `InteractiveLogger` class provides logging facilities
    for the console standard output. The logger shows
    a progress bar during training and evaluation flows and
    interactively display metric results as soon as they
    become available. The logger writes metric results after
    each training epoch, evaluation experience and at the
    end of the entire evaluation stream.

    .. note::
        To avoid an excessive amount of printed lines,
        this logger will **not** print results after
        each iteration. If the user is monitoring
        metrics which emit results after each minibatch
        (e.g., `MinibatchAccuracy`), only the last recorded
        value of such metrics will be reported at the end
        of the epoch.

    .. note::
        Since this logger works on the standard output,
        metrics producing images or more complex visualizations
        will be converted to a textual format suitable for
        console printing. You may want to add more loggers
        to your `EvaluationPlugin` to better support
        different formats.
    """

    def __init__(self):
        super().__init__(file=sys.stdout)
        self._pbar = None

    def before_training_epoch(self, strategy: 'BaseStrategy',
                              metric_values: List['MetricValue'], **kwargs):
        super().before_training_epoch(strategy, metric_values, **kwargs)
        self._progress.total = len(strategy.dataloader)

    def after_training_epoch(self, strategy: 'BaseStrategy',
                             metric_values: List['MetricValue'], **kwargs):
        self._end_progress()
        super().after_training_epoch(strategy, metric_values, **kwargs)

    def before_eval_exp(self, strategy: 'BaseStrategy',
                        metric_values: List['MetricValue'], **kwargs):
        super().before_eval_exp(strategy, metric_values, **kwargs)
        self._progress.total = len(strategy.dataloader)

    def after_eval_exp(self, strategy: 'BaseStrategy',
                       metric_values: List['MetricValue'], **kwargs):
        self._end_progress()
        super().after_eval_exp(strategy, metric_values, **kwargs)

    def after_training_iteration(self, strategy: 'BaseStrategy',
                                 metric_values: List['MetricValue'], **kwargs):
        self._progress.update()
        self._progress.refresh()
        super().after_training_iteration(strategy, metric_values, **kwargs)

    def after_eval_iteration(self, strategy: 'BaseStrategy',
                             metric_values: List['MetricValue'], **kwargs):
        self._progress.update()
        self._progress.refresh()
        super().after_eval_iteration(strategy, metric_values, **kwargs)

    @property
    def _progress(self):
        if self._pbar is None:
            self._pbar = tqdm(leave=True, position=0, file=sys.stdout)
        return self._pbar

    def _end_progress(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

class TqdmWriteInteractiveLogger(InteractiveLogger):
    """
    Allows to print out stats to console while updating
    progress bar whitout breaking it.
    """
    def __init__(self, log_every: int=1):
        super().__init__()
        self.log_every = log_every
        self.step_counter:int = 0

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