from avalanche.benchmarks.scenarios.generic_definitions import Experience
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.benchmarks.generators.rl_benchmark_generators import gym_benchmark_generator
from avalanche.training.plugins.reinforcement_learning import *
import torch

if __name__ == "__main__":
    device = torch.device('cuda:0')

    scenario = gym_benchmark_generator(['CartPole-v0', 'CartPole-v1'])
    # model = SimpleMLP(num_classes=10)
    # TODO: allow only plugins instead of strategies? or define model
    #  first and then pass it..? Strategies tho don't implement callbacks I guess
    # cl_strategy = Naive(model, optim, )
    strategy = RLStrategy('MlpPolicy', [scenario.envs[0]], 'dqn', None, per_experience_episodes=3, eval_mb_size=1, device=device, )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Env ", experience.env)
        print(isinstance(experience, Experience))
        strategy.train(experience)
        print('Training completed')

        # print('Computing accuracy on the whole test set')
        # results.append(cl_strategy.eval(scenario.test_stream))