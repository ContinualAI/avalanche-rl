from avalanche.training.strategies.reinforcement_learning import A2CStrategy, RLBaseStrategy
from avalanche.training.strategies.rl_utils import Array2Tensor
from avalanche.models.actor_critic import ActorCriticMLP
from avalanche.benchmarks.scenarios.generic_definitions import Experience
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from avalanche.training.strategies import Naive
from avalanche.benchmarks.generators.rl_benchmark_generators import gym_benchmark_generator
# from avalanche.training.plugins.reinforcement_learning import *
import torch
import gym
from torch.distributions import Categorical


def evaluate(model: torch.nn.Module):
    # TODO: this must be a module
    print("Evaluating agent")
    env = gym.make('CartPole-v1')
    env = Array2Tensor(env)
    obs = env.reset()
    done = False
    t = 0
    while not done:
        with torch.no_grad():
            _, logits = model(obs, compute_value=False)
        action = Categorical(logits=logits).sample().item()
        obs, _, done, _ = env.step(action)
        t += 1
    print("Steps performed:", t)


if __name__ == "__main__":
    device = torch.device('cuda:0')
    # TODO: benchmark should make Env parallel?
    scenario = gym_benchmark_generator(['CartPole-v0', 'CartPole-v1'])
    # CartPole setting
    model = ActorCriticMLP(4, 2, 1024)
    # cl_strategy = Naive(model, optim, )
    # strategy = RLStrategy('MlpPolicy', [scenario.envs[0]], 'dqn', None, per_experience_episodes=3, eval_mb_size=1, device=device, )

    optimizer = Adam(model.parameters())
    strategy = A2CStrategy(model, optimizer, per_experience_steps=1, )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Env ", experience.env)
        strategy.train(experience)
        print('Training completed')

        # print('Computing accuracy on the whole test set')
        # results.append(cl_strategy.eval(scenario.test_stream))

    evaluate(model)
