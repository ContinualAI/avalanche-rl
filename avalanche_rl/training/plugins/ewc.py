from collections import defaultdict
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict


class EWCPlugin(StrategyPlugin):
    """
    Elastic Weight Consolidation (EWC) plugin.
    EWC computes importance of each weight at the end of training on current
    experience. During training on each minibatch, the loss is augmented
    with a penalty which keeps the value of the current weights close to the
    value they had on previous experiences in proportion to their importance
    on that experience. Importances are computed with an additional pass on the
    training set. This plugin does not use task identities.
    """

    def __init__(self, ewc_lambda, mode='separate', decay_factor=None,
                 keep_importance_data=False):
        """
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience.
               `online` to keep a single penalty summed with a decay factor
               over all previous tasks.
        :param decay_factor: used only if mode is `online`.
               It specifies the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        """

        super().__init__()

        assert mode == 'separate' or mode == 'online', \
            'Mode must be separate or online.'

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor

        if self.mode == 'separate':
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        self.saved_params = defaultdict(list)
        self.importances = defaultdict(list)

    def before_backward(self, strategy, **kwargs):
        """
        Compute EWC penalty and add it to the loss.
        """

        if strategy.training_exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)

        if self.mode == 'separate':
            for experience in range(strategy.training_exp_counter):
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                        strategy.model.named_parameters(),
                        self.saved_params[experience],
                        self.importances[experience]):
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif self.mode == 'online':
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    strategy.model.named_parameters(),
                    self.saved_params[strategy.training_exp_counter],
                    self.importances[strategy.training_exp_counter]):
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError('Wrong EWC mode.')

        strategy.loss += self.ewc_lambda * penalty

    def after_training_exp(self, strategy, **kwargs):
        """
        Compute importances of parameters after each experience.
        """

        importances = self.compute_importances(strategy.model,
                                               strategy._criterion,
                                               strategy.optimizer,
                                               strategy.experience.dataset,
                                               strategy.device,
                                               strategy.train_mb_size)
        self.update_importances(importances, strategy.training_exp_counter)
        self.saved_params[strategy.training_exp_counter] = \
            copy_params_dict(strategy.model)
        # clear previuos parameter values
        if strategy.training_exp_counter > 0 and \
                (not self.keep_importance_data):
            del self.saved_params[strategy.training_exp_counter - 1]

    def compute_importances(self, model, criterion, optimizer,
                            dataset, device, batch_size):
        """
        Compute EWC importance matrix for each parameter
        """

        model.train()

        # list of list
        importances = zerolike_params_dict(model)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for i, (x, y, task_labels) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(),
                                          importances):
                assert (k1 == k2)
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))

        return importances

    @torch.no_grad()
    def update_importances(self, importances, t):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == 'separate' or t == 0:
            self.importances[t] = importances
        elif self.mode == 'online':
            for (k1, old_imp), (k2, curr_imp) in \
                    zip(self.importances[t - 1], importances):
                assert k1 == k2, 'Error in importance computation.'
                self.importances[t].append(
                    (k1, (self.decay_factor * old_imp + curr_imp)))

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")


class EWCRL(EWCPlugin):
    """
    Elastic Weight Consolidation (EWC) plugin for Reinforcement Learning, as presented
    in the original paper "Overcoming catastrophic forgetting in neural networks".
    As opposed to the non-rl version, importances are computed by sampling from a 
    ReplayMemory a pre-defined number of times and then running those batches through
    the network. 
    """

    def __init__(
            self, ewc_lambda, replay_memory: 'ReplayMemory', mode='separate', fisher_update_steps: int = 10,
            batch_size:int=32, start_ewc_after_steps: int = 0, start_ewc_after_experience: int = 1,
            decay_factor=None, keep_importance_data=False):
        """
            :param ewc_lambda: hyperparameter to weigh the penalty inside the total
                loss. The larger the lambda, the larger the regularization.
            :param replay_memory: the replay memory to sample from.
            :param batch_size: size of batches sampled during importance computation.
            :param mode: `separate` to keep a separate penalty for each previous
                experience.
                `online` to keep a single penalty summed with a decay factor
                over all previous tasks.
            :param fisher_update_steps: How many times batches are sampled from the ReplayMemory 
                during computation of the Fisher importance. Defaults to 10.
            :param start_ewc_after_steps: Start computing importances and adding penalty only after this many steps. Defaults to 0.
            :param start_ewc_after_experience: Start computing importances and adding penalty only after this many experiences. Defaults to 0.
            :param decay_factor: used only if mode is `online`.
                It specifies the decay term of the importance matrix.
            :param keep_importance_data: if True, keep in memory both parameter
                    values and importances for all previous task, for all modes.
                    If False, keep only last parameter values and importances.
                    If mode is `separate`, the value of `keep_importance_data` is
                    set to be True.
        """
        super().__init__(ewc_lambda, mode=mode, decay_factor=decay_factor,
                         keep_importance_data=keep_importance_data)
        self.fisher_updates_per_step = fisher_update_steps
        self.ewc_start_timestep = start_ewc_after_steps
        self.ewc_start_exp = start_ewc_after_experience
        self.memory = replay_memory
        self.batch_size = batch_size

    def after_training_exp(self, strategy: 'RLBaseStrategy', **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        # compute fisher information on task switch
        importances = self.compute_importances(strategy.model,
                                               strategy,
                                               strategy.optimizer,
                                               )

        self.update_importances(importances, strategy.training_exp_counter)

        self.saved_params[strategy.training_exp_counter] = copy_params_dict(
            strategy.model)
        # clear previuos parameter values
        if strategy.training_exp_counter > 0 and \
                (not self.keep_importance_data):
            del self.saved_params[strategy.training_exp_counter - 1]

    def before_backward(self, strategy: 'RLBaseStrategy', **kwargs):
        # add fisher penalty only after X steps
        if strategy.timestep >= self.ewc_start_timestep and strategy.training_exp_counter >= self.ewc_start_exp:
            return super().before_backward(strategy, **kwargs)

    def compute_importances(
            self, model, strategy: 'RLBaseStrategy', optimizer):
        # compute importances sampling minibatches from a replay memory/buffer
        model.train()

        importances = zerolike_params_dict(model)
        from avalanche.training.strategies.reinforcement_learning import DQNStrategy
        for _ in range(self.fisher_updates_per_step):
            if isinstance(strategy, DQNStrategy):
                # in DQN loss sampling from replay memory happens inside
                strategy.update(None)
            else:
                # sample batch
                batch = self.memory.sample_batch(self.batch_size, strategy.device)
                strategy.update([batch])

            optimizer.zero_grad()
            strategy.loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(), importances):
                assert (k1 == k2)
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over number of batches 
        for _, imp in importances:
            imp /= float(self.fisher_updates_per_step)
        optimizer.zero_grad()

        return importances


ParamDict = Dict[str, Tensor]
EwcDataType = Tuple[ParamDict, ParamDict]
