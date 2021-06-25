import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical


class ActorCriticMLP(nn.Module):
    # adapted from https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCriticMLP, self).__init__()
        # these are actually 2 models in one

        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state: torch.Tensor, compute_policy=True, compute_value=True):
        value, policy_logits = None, None
        if compute_value:
            value = F.relu(self.critic_linear1(state))
            value = self.critic_linear2(value)
        if compute_policy:
            policy_logits = F.relu(self.actor_linear1(state))
            policy_logits = self.actor_linear2(policy_logits)

        return value, policy_logits

    @torch.no_grad()
    def get_action(self, observation: torch.Tensor):
        _, policy_logits = self(observation, compute_value=False)
        return Categorical(logits=policy_logits).sample()