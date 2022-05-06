import torch
import torch.nn as nn
from .rl_base_strategy import RLBaseStrategy, Timestep, TimestepUnit
from .buffers import Rollout
from torch.optim.optimizer import Optimizer
from typing import Union, Optional, Sequence, List
from avalanche_rl.training.plugins.strategy_plugin import RLPlugin
from torch.optim import Optimizer
from torch.distributions import Categorical
from avalanche_rl.training import default_rl_logger
from avalanche_rl.models.actor_critic import A2CModel


class A2CStrategy(RLBaseStrategy):

    def __init__(
            self, model: A2CModel, optimizer: Optimizer,
            per_experience_steps: Union[int, Timestep, List[Timestep]],
            max_steps_per_rollout: int = 5,
            value_criterion=nn.MSELoss(),
            device='cpu',
            plugins: Optional[Sequence[RLPlugin]] = [],
            eval_every: int = -1, eval_episodes: int = 1, 
            policy_loss_weight: float = 0.5,
            value_loss_weight: float = 0.5,
            evaluator=default_rl_logger, **kwargs):
        # multiple steps per rollout are supported through time dimension flattening
        # e.g. working with tensors of shape `n_envs`*`timesteps`x`obs_shape`
        super().__init__(
            model, optimizer, per_experience_steps=per_experience_steps,
            # only support max steps as to avoid getting rollouts of different length
            rollouts_per_step=-1,
            max_steps_per_rollout=max_steps_per_rollout,
            device=device, plugins=plugins,
            eval_every=eval_every, eval_episodes=eval_episodes, 
            evaluator=evaluator, **kwargs)

        for exp_step in self.per_experience_steps:
            exp_step.unit == TimestepUnit.STEPS, 'A2C only supports expressing training duration in steps not episodes'

        self.value_criterion = value_criterion
        self.ac_w = policy_loss_weight
        self.cr_w = value_loss_weight

    def sample_rollout_action(self, observations: torch.Tensor):
        """
            This will process a batch of observations and produce a batch
            of actions to better leverage GPU as in 'batched' A2C, as in
            `n_envs` x D -> `n_envs` x A.
        Args:
            observations (torch.Tensor): [description]

        Returns:
            [type]: [description]
        """
        # sample action from policy network
        with torch.no_grad():
            _, policy_logits = self._model_forward(
                self.model, observations, compute_value=False)
        # (alternative np.random.choice(num_outputs, p=np.squeeze(dist)))
        return Categorical(logits=policy_logits).sample().cpu().numpy()

    def update(self, rollouts: List[Rollout]):
        # perform gradient step(s) over gathered rollouts
        self.loss = 0.
        for rollout in rollouts:
            # move samples to device for processing and expect tensor of shape `timesteps`x`n_envs`xD`
            rollout = rollout.to(self.device)
            # print("Rollout Observation shape", rollout.observations.shape)
            values, policy_logits = self._model_forward(
                self.model, rollout.observations)
            # ~log(softmax(taken_action_logits))
            # print("Rollout Actions shape", rollout.actions.shape)
            # FIXME: remove view
            log_prob = Categorical(
                logits=policy_logits).log_prob(
                rollout.actions.view(-1,))
            # compute next states values
            next_values, _ = self._model_forward(
                self.model, rollout.next_observations, compute_policy=False)
            # mask terminal states values
            next_values[rollout.dones.view(-1,)] = 0.

            # Actor/Policy Loss Term in A2C: A(s_t, a_t) * grad log (pi(a_t|s_t))
            boostrapped_returns = rollout.rewards + self.gamma * next_values
            advantages = boostrapped_returns - values 
            # get advantages of taken actions a_t FIXME: this whill need view(-1,1)
            advantages = advantages.gather(dim=1, index=rollout.actions)
            # print("Rollout adv shape", advantages.shape, log_prob.shape, policy_logits.shape)
            policy_loss = -(advantages * log_prob).mean()

            # Value Loss Term: (R_t + gamma * V(S_{t+1}) - V(S_t))^2
            # value_loss = advantages.pow(2)
            value_loss = self.value_criterion(boostrapped_returns, values)

            # accumulate gradients for multi-rollout case
            self.loss += self.ac_w * policy_loss + self.cr_w * value_loss
