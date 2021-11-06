from typing import Dict, Tuple
from torch import Tensor
from avalanche.training.plugins.ewc import EWCPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from avalanche_rl.training.strategies.buffers import ReplayMemory
from avalanche_rl.training.strategies import RLBaseStrategy

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
        from avalanche_rl.training.strategies.dqn import DQNStrategy
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
