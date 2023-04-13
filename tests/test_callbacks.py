import torch
import pytest
import sys
from avalanche_rl.training.plugins.rl_plugins import RLStrategyPlugin
from avalanche_rl.models.dqn import MLPDeepQN
from avalanche_rl.benchmarks.rl_benchmark_generators \
    import gym_benchmark_generator 
from avalanche_rl.training.strategies.dqn import DQNStrategy
from avalanche.logging.text_logging import TextLogger
from torch.optim import Adam


class MockPlugin(RLStrategyPlugin):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.activated = [False for _ in range(20)]
        self.callbacks = [
            "before_training_exp",
            "before_training_iteration",
            "before_forward",
            "after_forward",
            "before_backward",
            "after_backward",
            "after_training_iteration",
            "before_update",
            "after_update",
            "after_training_exp",
            "before_eval",
            "before_eval_exp",
            "after_eval_exp",
            "after_eval",
            "before_eval_iteration",
            "before_eval_forward",
            "after_eval_forward",
            "after_eval_iteration",
            "before_rollout",
            "after_rollout"
        ]

    def before_training_exp(self, strategy, **kwargs):
        self.activated[0] = True

    def before_training_iteration(self, strategy, **kwargs):
        self.activated[1] = True

    def before_forward(self, strategy, **kwargs):
        self.activated[2] = True

    def after_forward(self, strategy, **kwargs):
        self.activated[3] = True

    def before_backward(self, strategy, **kwargs):
        self.activated[4] = True

    def after_backward(self, strategy, **kwargs):
        self.activated[5] = True

    def after_training_iteration(self, strategy, **kwargs):
        self.activated[6] = True

    def before_update(self, strategy, **kwargs):
        self.activated[7] = True

    def after_update(self, strategy, **kwargs):
        self.activated[8] = True

    def after_training_exp(self, strategy, **kwargs):
        self.activated[9] = True

    def before_eval(self, strategy, **kwargs):
        self.activated[10] = True

    def before_eval_exp(self, strategy, **kwargs):
        self.activated[11] = True

    def after_eval_exp(self, strategy, **kwargs):
        self.activated[12] = True

    def after_eval(self, strategy, **kwargs):
        self.activated[13] = True

    def before_eval_iteration(self, strategy, **kwargs):
        self.activated[14] = True

    def before_eval_forward(self, strategy, **kwargs):
        self.activated[15] = True

    def after_eval_forward(self, strategy, **kwargs):
        self.activated[16] = True

    def after_eval_iteration(self, strategy, **kwargs):
        self.activated[17] = True

    def before_rollout(self, strategy, **kwargs):
        self.activated[18] = True

    def after_rollout(self, strategy, **kwargs):
        self.activated[19] = True


def test_callback_reachability():
    # Check that all the callbacks are called during
    # training and test loops.
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    scenario = gym_benchmark_generator(
        ['CartPole-v1'],
        n_parallel_envs=1, eval_envs=['CartPole-v1'], n_experiences=1)

    # CartPole setting
    model = MLPDeepQN(input_size=4, hidden_size=1024, 
                      n_actions=2, hidden_layers=2)
    # print("Model", model)

    # DQN learning rate
    optimizer = Adam(model.parameters(), lr=1e-3)
    plug = MockPlugin()

    strategy = DQNStrategy(model, optimizer, 100, batch_size=32,
                           exploration_fraction=.2, rollouts_per_step=10,
                           replay_memory_size=1000, updates_per_step=10,
                           replay_memory_init_size=1000, double_dqn=False,
                           target_net_update_interval=10, eval_every=100,
                           eval_episodes=10, plugins=[plug],
                           device=device, max_grad_norm=None)

    strategy.evaluator.loggers = [TextLogger(sys.stderr)]

    for experience in scenario.train_stream:
        strategy.train(experience, scenario.eval_stream)
    strategy.eval(scenario.eval_stream)

    # for p, v in zip(plug.callbacks, plug.activated):
    #     print(f"{p:30s}:\t{v}")

    assert all(plug.activated)
