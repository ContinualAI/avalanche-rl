import torch.nn as nn
import torch
import torch.nn.functional as F
from .simple_mlp import SimpleMLP


class DQNModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(x: torch.Tensor, task_label=None):
        raise NotImplementedError()

    @torch.no_grad()
    def get_action(self, observation: torch.Tensor, task_label=None):
        q_values = self(observation, task_label=task_label)
        return torch.argmax(q_values, dim=1).cpu().int().numpy()


class MLPDeepQN(DQNModel):
    """
    Simple Action-Value MLP for DQN.
    """

    def __init__(
            self, input_size: int, hidden_size: int, n_actions: int,
            hidden_layers: int = 1):
        super().__init__()
        # it does use dropout
        self.dqn = SimpleMLP(
            num_classes=n_actions, input_size=input_size,
            hidden_size=hidden_size, hidden_layers=hidden_layers, dropout=False)
    
    def forward(self, x: torch.Tensor, task_label=None):
        return self.dqn(x)
        


class ConvDeepQN(DQNModel):
    # network architecture from Mnih et al 2015 - "Human-level Control Through Deep Reinforcement Learning"
    def __init__(self, input_channels, image_shape, n_actions, batch_norm=False):
        super(ConvDeepQN, self).__init__()
        # 4x84x84 input in original paper
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc = nn.Sequential(
            nn.Linear(
                self._compute_flattened_shape(
                    (input_channels, image_shape[0],
                     image_shape[1])),
                512),
            nn.ReLU(),
            nn.Linear(512, n_actions))

    def forward(self, x, task_label=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # feed to linear layer
        x = x.flatten(1)
        return self.fc(x)

    def _compute_flattened_shape(self, input_shape):
        x = torch.zeros(input_shape)
        x = x.unsqueeze(0)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        print("Size of flattened input to fully connected layer:", x.flatten().shape)
        return x.squeeze(0).flatten().shape[0]