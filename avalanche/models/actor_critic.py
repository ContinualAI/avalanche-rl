import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
from typing import List, Union

class ActorCriticMLP(nn.Module):

    def __init__(self, num_inputs, num_actions, actor_hidden_sizes: Union[int, List[int]]=[64, 64], critic_hidden_sizes: Union[int, List[int]]=[64, 64], activation_type:str='relu'):
        super(ActorCriticMLP, self).__init__()
        # these are actually 2 models in one
        if type(actor_hidden_sizes) is int:
            actor_hidden_sizes = [actor_hidden_sizes]
        if type(critic_hidden_sizes) is int:
            critic_hidden_sizes = [critic_hidden_sizes]
        assert len(critic_hidden_sizes) and len(actor_hidden_sizes)
        if activation_type == 'relu':
            act = nn.ReLU()
        elif activation_type == 'tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation type {activation_type}")

        critic = [nn.Linear(critic_hidden_sizes[i], critic_hidden_sizes[i+1]) for i in range(1, len(critic_hidden_sizes)-1)]
        actor = [nn.Linear(actor_hidden_sizes[i], actor_hidden_sizes[i+1]) for i in range(1, len(actor_hidden_sizes)-1)]
        
        # self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.critic = []
        for layer in [nn.Linear(num_inputs, critic_hidden_sizes[0])]+critic:
            self.critic.append(layer)
            self.critic.append(act)
        self.critic.append(nn.Linear(critic_hidden_sizes[-1], num_actions))
        self.critic = nn.Sequential(*self.critic)

        self.actor = []
        for layer in [nn.Linear(num_inputs, actor_hidden_sizes[0])]+actor:
            self.actor.append(layer)
            self.actor.append(act)
        self.actor.append(nn.Linear(actor_hidden_sizes[-1], num_actions))
        self.actor = nn.Sequential(*self.actor)

    
    def forward(self, state: torch.Tensor, compute_policy=True, compute_value=True):
        value, policy_logits = None, None
        if compute_value:
            value = self.critic(state)
        if compute_policy:
            policy_logits = self.actor(state)

        return value, policy_logits

    @torch.no_grad()
    def get_action(self, observation: torch.Tensor):
        _, policy_logits = self(observation, compute_value=False)
        return Categorical(logits=policy_logits).sample()