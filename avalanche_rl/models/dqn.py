import torch.nn as nn
import torch
import torch.nn.functional as F
from avalanche.models.simple_mlp import SimpleMLP


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
        # disable dropout by default
        self.dqn = SimpleMLP(
            num_classes=n_actions, input_size=input_size,
            hidden_size=hidden_size, hidden_layers=hidden_layers, drop_rate=0.)
    
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


class EWCConvDeepQN(DQNModel):
    """Model used in the original EWC paper https://arxiv.org/abs/1612.00796.
        It is a variant of the original DQN with added task-specific biases and gains. 
    """
    def __init__(self, input_channels, image_shape, n_actions, n_tasks, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias=bias)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, bias=bias)
        shapes = self._compute_shapes(
            (input_channels, image_shape[0], image_shape[1]))

        # bias/gain are game-specific and are initialized as in the paper
        for layer in range(1, 4):
            for task in range(n_tasks):
                setattr(self, f'bias{layer}_{task}', nn.parameter.Parameter(torch.zeros(*shapes[layer-1])))
                setattr(self, f'gain{layer}_{task}', nn.parameter.Parameter(torch.ones(*shapes[layer-1])))

        # fully connected part
        self.l1 = nn.Linear(shapes[-1], 1024, bias=bias)
        self.l2 = nn.Linear(1024, n_actions, bias=bias)

        # linear layers biases & gains
        fc_sizes = [1024, n_actions]
        for layer in range(1, 3):
            for task in range(n_tasks):
                setattr(self, f'bias_l{layer}_{task}', nn.parameter.Parameter(torch.zeros(fc_sizes[layer-1],)))
                setattr(self, f'gain_l{layer}_{task}', nn.parameter.Parameter(torch.ones(fc_sizes[layer-1])))


    def forward(self, x: torch.Tensor, task_label=None) -> torch.Tensor:
        # biases and gains are game-specific: select them using task label
        for i in range(1, 4):
            x = getattr(self, f'conv{i}')(x)
            task_bias = getattr(self, f'bias{i}_{task_label}')
            gain = getattr(self, f'gain{i}_{task_label}')
            # print('conv shape', x.shape, task_bias.shape)
            x += task_bias
            x *= gain
            # torch.add(x, bias, alpha=gains)?
            x = F.relu(x)

        # feed to fc layer
        x = x.flatten(1)

        x = self.l1(x)
        x += getattr(self, f'bias_l1_{task_label}')
        x *= getattr(self, f'gain_l1_{task_label}')
        x = F.relu(x)

        x = self.l2(x)
        x += getattr(self, f'bias_l2_{task_label}')
        x *= getattr(self, f'gain_l2_{task_label}')

        return x

    def _compute_shapes(self, input_shape):
        # returns activation maps sizes at each layer for adding biases & gains
        x = torch.zeros(input_shape)
        x = x.unsqueeze(0)
        with torch.no_grad():
            x = self.conv1(x)
            s1 = x.shape[2:]
            x = self.conv2(x)
            s2 = x.shape[2:]
            x = self.conv3(x)
            s3 = x.shape[2:]
        return s1, s2, s3, x.squeeze(0).flatten().shape[0]