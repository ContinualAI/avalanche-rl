import torch
import torch.nn as nn
from .rl_base_strategy import RLBaseStrategy, Timestep, TimestepUnit
from .buffers import Rollout
from torch.optim.optimizer import Optimizer
from typing import Union, Optional, Sequence, List
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from torch.optim import Optimizer
from torch.distributions import Categorical


class A2CStrategy(RLBaseStrategy):

    def __init__(
            self, model: nn.Module, optimizer: Optimizer,
            per_experience_steps: Union[int, Timestep],
            max_steps_per_rollout: int = 1, value_criterion=nn.MSELoss(),
            discount_factor: float = 0.99, device='cpu',
            updates_per_step: int = 1,
            plugins: Optional[Sequence[StrategyPlugin]] = [],
            eval_every=-1, policy_loss_weight: float = 0.5,
            value_loss_weight: float = 0.5,):
        super().__init__(
            model, optimizer, per_experience_steps=per_experience_steps,
            # here we make sure we can only do steps not episodes
            rollouts_per_step=-1,
            max_steps_per_rollout=max_steps_per_rollout,
            updates_per_step=updates_per_step, device=device, plugins=plugins,
            discount_factor=discount_factor, eval_every=eval_every)
        assert self.per_experience_steps.unit == TimestepUnit.STEPS, 'A2C only supports expressing training duration in steps not episodes'

        # TODO: 'dataloader' calls with pre-processing env wrappers 
        self.model = model
        self.optimizer = optimizer
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
            _, policy_logits = self.model(observations, compute_value=False)
        # (alternative np.random.choice(num_outputs, p=np.squeeze(dist)))
        return Categorical(logits=policy_logits).sample().cpu().numpy()

    def update(self, rollouts: List[Rollout], n_update_steps: int):
        # perform gradient step(s) over batch of gathered rollouts
        self.loss = 0.
        for rollout in rollouts:
            # move samples to device for processing
            rollout = rollout.to(self.device)
            # print("Rollout Observation shape", rollout.observations.shape)
            values, policy_logits = self.model(rollout.observations)
            # ~log(softmax(action_logits))
            # print("Rollout Actions shape", rollout.actions.shape)
            log_prob = Categorical(
                logits=policy_logits).log_prob(
                rollout.actions)
            # compute next states values
            next_values, _ = self.model(
                rollout.next_observations, compute_policy=False)
            # mask terminal states values
            next_values[rollout.dones] = 0.

            # print("Rollout Rewards shape", rollout.rewards.shape)
            # Actor/Policy Loss Term in A2C: A(s_t, a_t) * grad log (pi(a_t|s_t))
            boostrapped_returns = rollout.rewards + self.gamma * next_values
            advantages = boostrapped_returns - values 
            policy_loss = -(advantages * log_prob).mean()

            # Value Loss Term: R_t + gamma * V(S_{t+1}) - V(S_t)
            # value_loss = advantages.pow(2)
            value_loss = self.value_criterion(boostrapped_returns, values)

            # accumulate gradients for multi-rollout case
            self.loss += self.ac_w * policy_loss + self.cr_w * value_loss
