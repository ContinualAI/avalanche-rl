from avalanche.logging.interactive_logging import TqdmWriteInteractiveLogger
from avalanche.evaluation.metrics.reward import EpLenghtPluginMetric, RewardPluginMetric
from avalanche.logging.tensorboard_logger import TensorboardLogger
from avalanche.training.strategies.reinforcement_learning import A2CStrategy, RLBaseStrategy, DQNStrategy
from avalanche.training.strategies.reinforcement_learning.utils import Array2Tensor
from avalanche.models.actor_critic import ActorCriticMLP
from avalanche.models.dqn import ConvDeepQN, MLPDeepQN
from avalanche.benchmarks.scenarios.generic_definitions import Experience
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from avalanche.training.strategies import Naive
from avalanche.benchmarks.generators.rl_benchmark_generators import gym_benchmark_generator
# from avalanche.training.plugins.reinforcement_learning import *
import torch
import gym
from torch.distributions import Categorical
import numpy as np
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import cpu_usage_metrics, ram_usage_metrics, gpu_usage_metrics


def evaluate(model: torch.nn.Module, n_episodes=10, device=torch.device('cpu')):
    # TODO: this must be a module
    print("Evaluating agent")
    env = gym.make('CartPole-v1')
    env = Array2Tensor(env)
    rewards, lengths = [], []
    model = model.to(device)
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        t = 0
        while not done:
            action = model.get_action(obs.unsqueeze(0).to(device))
            obs, r, done, _ = env.step(action.item())
            rewards.append(r)
            t += 1
        lengths.append(t)
    return np.mean(rewards), np.std(rewards), np.mean(lengths)


if __name__ == "__main__":
    # FIXME:
    # tensorboard_logger = TensorboardLogger('../tensorboard_logs')

    # eval_plugin = EvaluationPlugin(
    #    cpu_usage_metrics(
    #        minibatch=True, epoch=True,
    #        experience=True, stream=True),
    #    ram_usage_metrics(
    #        every=0.5, minibatch=True, epoch=True,
    #        experience=True, stream=True),
    #    gpu_usage_metrics(
    #        0, every=0.5, minibatch=True, epoch=True,
    #        experience=True, stream=True),
    #    RewardPluginMetric(
    #        window_size=1000,
    #        stats=['mean', 'max', 'std']),
    #    EpLenghtPluginMetric(
    #        window_size=1000,
    #        stats=['mean', 'max', 'std']),
    #    loggers=[
    #             TqdmWriteInteractiveLogger(
    #                 log_every=10),
    #             tensorboard_logger])

    device = torch.device('cpu')
    # device = torch.device('cuda:0')

    # FIXME: multi env is still bugged, single env works just fine
    # works well if you use VectorizedEnv with 1 parallel env tho..
    scenario = gym_benchmark_generator(
        ['CartPole-v1'],
        n_parallel_envs=1, eval_envs=['CartPole-v1'])
    # scenario = gym_benchmark_generator(['MountainCar-v0'], n_parallel_envs=1)

    # CartPole setting
    model = ActorCriticMLP(4, 2, 1024, 1024)
    # model = ActorCriticMLP(2, 3, 512, 512)
    print("Model", model)
    # model = MLPDeepQN(input_size=4, hidden_size=128,
    #   n_actions=2, hidden_layers=2)
    # cl_strategy = Naive(model, optim, )
    # strategy = RLStrategy('MlpPolicy', [scenario.envs[0]], 'dqn', None, per_experience_episodes=3, eval_mb_size=1, device=device, )
    optimizer = Adam(model.parameters(), lr=1e-4)

    strategy = A2CStrategy(
        model, optimizer, per_experience_steps=100000, max_steps_per_rollout=5,
        device=device, eval_every=10000, eval_episodes=100)
    # strategy = DQNStrategy(
    # model, optimizer, 1000, batch_size=32, exploration_fraction=.2, rollouts_per_step=10,
    # replay_memory_size=10000, updates_per_step=10, replay_memory_init_size=500, double_dqn=True,
    # target_net_update_interval=10, polyak_update_tau=1.)

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Env ", experience.env)
        strategy.train(experience, scenario.test_stream)
        print('Training completed')

        print("Test stream", [e.environment for e in scenario.test_stream])
        print('Computing accuracy on the whole test set')
        results.append(strategy.eval(scenario.test_stream))

    rmean, rstd, lengths = evaluate(model, n_episodes=100, device=device)
    print(f"Reward mean/std: {rmean}, {rstd}. Episode lengths: {lengths}")
