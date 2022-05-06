from avalanche_rl.training.strategies import DQNStrategy
from avalanche_rl.models.dqn import MLPDeepQN
from torch.optim import Adam
from avalanche_rl.benchmarks.generators.rl_benchmark_generators import gym_benchmark_generator
import torch

if __name__ == "__main__":
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    scenario = gym_benchmark_generator(
        ['CartPole-v1'],
        n_parallel_envs=1, eval_envs=['CartPole-v1'], n_experiences=1)

    # CartPole setting
    model = MLPDeepQN(input_size=4, hidden_size=1024,
                      n_actions=2, hidden_layers=2)
    print("Model", model)
    # DQN learning rate
    optimizer = Adam(model.parameters(), lr=1e-3)

    strategy = DQNStrategy(model, optimizer, 100, batch_size=32, exploration_fraction=.2, rollouts_per_step=10,
                           replay_memory_size=1000, updates_per_step=10, replay_memory_init_size=1000, double_dqn=False,
                           target_net_update_interval=10, eval_every=100, eval_episodes=10, 
                           device=device, max_grad_norm=None)

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Env ", experience.env)
        print("Current Task", experience.task_label, type(experience.task_label))
        strategy.train(experience, scenario.eval_stream)

    print('Training completed')
    eval_episodes = 100
    print(f"\nEvaluating on {eval_episodes} episodes!")
    print(strategy.eval(scenario.eval_stream))
