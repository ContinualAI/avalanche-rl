import os
from gym.envs.registration import register

__version__ = "0.0.1"

_dataset_add = None


def _avdataset_radd(self, other, *args, **kwargs):
    from avalanche.benchmarks.utils import AvalancheDataset
    global _dataset_add
    if isinstance(other, AvalancheDataset):
        return NotImplemented

    return _dataset_add(self, other, *args, **kwargs)


def _avalanche_monkey_patches():
    from torch.utils.data.dataset import Dataset
    global _dataset_add
    _dataset_add = Dataset.__add__
    Dataset.__add__ = _avdataset_radd


_avalanche_monkey_patches()

# Register continual version of gym envs
register(
    id='CCartPole-v1',
    entry_point='avalanche.envs.classic_control:ContinualCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='CMountainCar-v0',
    entry_point='avalanche.envs.classic_control:ContinualMountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='CAcrobot-v1',
    entry_point='avalanche.envs.classic_control:ContinualAcrobotEnv',
    reward_threshold=-100.0,
    max_episode_steps=500,
)


# Hotfix for MNIST download issue
# def _adapt_dataset_urls():
#     from torchvision.datasets import MNIST
#     prev_mnist_urls = MNIST.resources
#     new_resources = [
#         ('https://storage.googleapis.com/cvdf-datasets/mnist/'
#          'train-images-idx3-ubyte.gz', prev_mnist_urls[0][1]),
#         ('https://storage.googleapis.com/cvdf-datasets/mnist/'
#          'train-labels-idx1-ubyte.gz', prev_mnist_urls[1][1]),
#         ('https://storage.googleapis.com/cvdf-datasets/mnist/'
#          't10k-images-idx3-ubyte.gz', prev_mnist_urls[2][1]),
#         ('https://storage.googleapis.com/cvdf-datasets/mnist/'
#          't10k-labels-idx1-ubyte.gz', prev_mnist_urls[3][1])
#     ]
#     MNIST.resources = new_resources
#
#
# _adapt_dataset_urls()
