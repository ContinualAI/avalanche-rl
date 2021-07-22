import torch
import torch.nn as nn
import numpy as np
from .rl_base_strategy import RLBaseStrategy, Timestep
from .buffers import Rollout, ReplayMemory
from torch.optim.optimizer import Optimizer
from typing import Union, Optional, Sequence, List
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from torch.optim import Optimizer
import copy
import random
from avalanche.training import default_rl_logger


class DQNStrategy(RLBaseStrategy):

    def __init__(
            self, model: nn.Module, optimizer: Optimizer,
            per_experience_steps: Union[int, Timestep, List[Timestep]], 
            rollouts_per_step: int = 8,  # how often do you perform an update step
            max_steps_per_rollout: int = -1,
            replay_memory_size: int = 10000,
            replay_memory_init_size: int = 5000,
            updates_per_step=1,
            criterion=nn.SmoothL1Loss(), 
            batch_size: int = 32,
            initial_epsilon: float = 1.0,
            final_epsilon: float = 0.05,
            exploration_fraction: float = 0.1,
            double_dqn: bool = True,
            target_net_update_interval: Union[int, Timestep] = 10000,
            polyak_update_tau: float = 1.,  # set to 1. to hard copy
            discount_factor: float = 0.99,
            device='cpu',
            plugins: Optional[Sequence[StrategyPlugin]] = [],
            eval_every: int = -1, eval_episodes: int = 1, evaluator=default_rl_logger):
        super().__init__(model, optimizer, per_experience_steps,
                         criterion=criterion, rollouts_per_step=rollouts_per_step,
                         max_steps_per_rollout=max_steps_per_rollout,
                         updates_per_step=updates_per_step, device=device, plugins=plugins,
                         discount_factor=discount_factor, eval_every=eval_every,
                         eval_episodes=eval_episodes, evaluator=evaluator)
        if type(target_net_update_interval) is int:
            target_net_update_interval:Timestep = Timestep(target_net_update_interval)
        for exp_step in self.per_experience_steps:
            assert target_net_update_interval.unit == exp_step.unit, "You must express the target network interval using the same unit as the training lenght"

        self.replay_memory: ReplayMemory = None
        self.replay_init_size = replay_memory_init_size
        self.replay_size = replay_memory_size
        self.batch_dim = batch_size
        self.double_dqn = double_dqn
        self.target_net_update_interval: Timestep = target_net_update_interval
        self.polyak_update_tau = polyak_update_tau
        assert initial_epsilon >= final_epsilon, "Initial epsilon value must be greater or equal than final one"

        self._init_eps = initial_epsilon
        self.eps = initial_epsilon
        self.final_eps = final_epsilon
        self.expl_fr = exploration_fraction

        # initialize target network
        self.target_net = copy.deepcopy(self.model)

    def _update_epsilon(self, experience_timestep: int):
        """
            Linearly decrease exploration rate `self.eps` up to `self.final_eps` in a 
            fraction of the total timesteps (`exploration_fraction`).
            It will reset to `self._init_eps` on new experience.
        """
        # TODO: log this values
        new_value = self._init_eps - experience_timestep * self.eps_decay
        self.eps = new_value if new_value > self.final_eps else self.final_eps

    def _update_target_network(self, timestep: int):
        # copy over network parameter to fixed target net
        if timestep > 0 and timestep % self.target_net_update_interval.value:
            # from stable baseline 3 enhancement https://github.com/DLR-RM/stable-baselines3/issues/93
            with torch.no_grad():
                # all done in-place for efficiency
                for param, target_param in zip(
                        self.model.parameters(),
                        self.target_net.parameters()):
                    target_param.data.mul_(1-self.polyak_update_tau)
                    torch.add(target_param.data, param.data,
                              alpha=self.polyak_update_tau,
                              out=target_param.data)

    def before_training_exp(self, **kwargs):
        # compute linear decay rate from specified fraction and specified timestep unit 
        # for this experience (supports different number of steps per experience)
        self.eps_decay = (self._init_eps - self.final_eps) / (self.expl_fr * self.current_experience_steps.value)

        # initialize replay memory with collected data before first experience, taking into account multiple workers
        rollouts = self.rollout(
            self.environment, n_rollouts=-1, max_steps=self.replay_init_size //
            self.n_envs)
        if self.replay_memory is None:
            self.replay_memory = ReplayMemory(
                size=self.replay_size, n_envs=self.n_envs)

        self.replay_memory.add_rollouts(rollouts)
        return super().before_training_exp()

    def before_rollout(self, **kwargs):
        # update exploration rate
        self._update_epsilon(self.timestep)
        # update fixed target network
        self._update_target_network(self.timestep)

        return super().before_rollout(**kwargs)

    def after_rollout(self, **kwargs):
        # add collected rollouts to replay memory
        self.replay_memory.add_rollouts(self.rollouts)
        return super().after_rollout(**kwargs)

    def sample_rollout_action(self, observations: torch.Tensor):
        """
            Generate action following epsilon-greedy strategy in which we either sample
            a random action with probability epsilon or we exploit current Q-value derived
            policy by taking the action with greatest Q-value.

        Args:
            observations (torch.Tensor): Observation coming from Env on previous step, 
            of shape `self.n_envs` x obs_shape.
        """ 
        # all actors interacting with environment either exploit or explore
        if random.random() > self.eps:
            # exploitation
            with torch.no_grad():
                q_values = self.model(observations)
                actions = torch.argmax(
                    q_values, dim=1).cpu().type(
                    torch.int64).numpy()
        else:
            actions = [
                self.environment.action_space.sample()
                for _ in range(self.n_envs)]
            actions = np.asarray(actions, dtype=np.int64)
        # actors run on cpu, return numpy array
        return actions

    @torch.no_grad()
    def _compute_next_q_values(self, batch: Rollout):
        # Compute next state q values using fixed target net
        next_q_values = self.target_net(batch.next_observations)

        if self.double_dqn:
            # Q'(s', argmax_a' Q(s', a') ):
            # use model to select the action with maximal value (follow greedy policy with current weights)
            max_actions = torch.argmax(
                self.model(batch.next_observations), dim=1)
            # evaluate q value of that action using fixed target network
            # select max actions, one per batch element
            next_q_values = next_q_values[torch.arange(
                next_q_values.shape[0]), max_actions]
            # equal to torch.gather(next_q_values, dim=1, index=max_actions.unsqueeze(-1))
        else:
            # get values corresponding to highest q value actions
            next_q_values, _ = next_q_values.max(dim=1)

        return next_q_values

    def update(self, rollouts: List[Rollout], n_update_steps: int):
        self.loss = 0.
        for _ in range(n_update_steps):
            # sample batch of steps/experiences from memory
            batch = self.replay_memory.sample_batch(self.batch_dim, self.device)

            # compute q values prediction for whole batch: Q(s, a)
            q_pred = self.model(batch.observations)
            # print('obs shape', batch.observations.shape,'act', batch.actions.shape, 'q pred', q_pred.shape)

            # condition on taken actions (select performed actions' q-values)
            q_pred = torch.gather(
                q_pred, dim=1, index=batch.actions)

            # compute target Q value: Q*(s, a) = R_t + gamma * max_{a'} Q(s', a') 
            next_q_values = self._compute_next_q_values(batch)
            # print('q next', next_q_values.shape, batch.rewards.shape, batch.dones.shape, 'q pred', q_pred.shape)
            # mask terminal states only after max q value action has been selected
            q_target = batch.rewards + self.gamma * \
                (1 - batch.dones.int()) * next_q_values.unsqueeze(-1)

            self.loss += self._criterion(q_pred, q_target)

            # TODO: gradient norm clipping?
