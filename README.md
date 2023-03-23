<div align="center">
    
# Avalanche RL: an End-to-End Library for Continual Reinforcement Learning
<!-- # Avalanche: an End-to-End Library for Continual Learning -->
**[Avalanche Website](https://avalanche.continualai.org)** | **[Paper](https://arxiv.org/abs/2202.13657)**

</div>

<p align="center">
    <img src="https://www.dropbox.com/s/90thp7at72sh9tj/avalanche_logo_with_clai.png?raw=1"/>
</p>

**Avalanche RL** is a Pytorch-based framework building upon ContinualAI's [**Avalanche**](https://github.com/ContinualAI/avalanche) with the goal of extending its capabilities to Continual Reinforcement Learning (*CRL*), *bootstrapping* from the work done on Super/Unsupervised Continual Learning.

It should support all environments sharing the `gym.Env` interface, handle stream of experiences, provide strategies for RL algorithms and enable fast prototyping through an extremely flexible and customizable API. 

The core structure and design principles of Avalanche are to remain untouched to easen the learning curve for all continual learning practitioners, so we still work with the same modules you can find in avl:
- [Benchmarks](avalanche_rl/benchmarks) for managing data and stream of data.
- [Training](avalanche_rl/training) for model training making use of extensible strategies.
- [Evaluation](avalanche_rl/evaluation) to evaluate the agent on consistent metrics.
- [Extras](avalanche_rl/extras) for general utils and building blocks.
- [Models](avalanche_rl/models) contains commonly used model architectures.
- [Logging](avalanche_rl/logging) for logging metrics during training/evaluation.

Head over to [Avalanche Website](https://avalanche.continualai.org) to learn more if these concepts sound unfamiliar to you!

## Features
___
Features added so far in this fork can be summarized and grouped by module.
<p align="center">
    <img src="assets/diagram.png"/>
</p>

### *Benchmarks*
[RLScenario](https://github.com/NickLucche/avalanche/blob/master/avalanche/benchmarks/rl_benchmark.py) introduces a Benchmark for RL which augments each experience with an 'Environment' (defined through [OpenAI `gym.Env` interface](https://github.com/openai/gym/blob/120e21cd75db36cce241f1b3a23184d3876c9753/gym/core.py#L8)) effectively implementing a "stream of environments" with which the agent can interact to generate data and learn from that interaction during each experience. This concept models the way experiences in the supervised CL context are translated to CRL, moving away from the concept of Dataset toward a dynamic interaction through which data is generated.

[RL Benchmark Generators](https://github.com/NickLucche/avalanche/blob/master/avalanche/benchmarks/generators/rl_benchmark_generators.py) allow to build these streams of experiences seamlessly, supporting:
 - Any sequence of `gym.Env` environments through `gym_benchmark_generator`, which returns a `RLScenario` from a list of environments ids (e.g. `["CartPole-v1", "MountainCar-v0", ..]`) with access to a train and test stream just like in Avalanche. It also supports sampling a random number of environments if you wanna get wild with your experiments.
 - Atari 2600 games through `atari_benchmark_generator`, taking care of common Wrappers (e.g. frame stacking) for these environments to get you started even more quickly.
 - [Habitat](https://github.com/facebookresearch/habitat-sim/), more on this later.

 ### *Training*
 [RLBaseStrategy]() is the super-class of all RL algorithms, augmenting `BaseStrategy` with RL specific callbacks while still making use of all major features such as plugins, logging and callbacks.
 Inspired by the amazing [`stable-baselines-3`](https://github.com/DLR-RM/stable-baselines3), it supports both on and off-policy algorithms under a common API defined as a 'rollouts phase' (data gathering) followed by an 'update phase', whose specifics are implemented by subclasses (RL algorithms).

 Algorithms are added to the framework by subclassing `RLBaseStrategy` and implementing specific callbacks. You can check out [this implementation of A2C](https://github.com/NickLucche/avalanche/blob/master/avalanche/training/strategies/reinforcement_learning/actor_critic.py) in under 50 lines of actual code including the `update` step and the action sampling mechanism.
 Currently only A2C and DQN+DoubleDQN algorithms have been implemented, including various other "utils" such as Replay Buffer.

 Training with multiple agent is supported through [`VectorizedEnv`](https://github.com/NickLucche/avalanche/blob/master/avalanche/training/strategies/reinforcement_learning/vectorized_env.py), leveraging [Ray](https://ray.io/) for parallel and potentially distributed execution of multiple environment interactions.   

 ### Evaluation
 New metrics have been added to keep track of rewards, episodes length and any kind of scalar value (such as Epsilon Greedy 'eps') during experiments. Metrics are kept track of using a moving averaged window, useful for smoothing out fluctuations and recording standard deviation and max values reached.  
 ### Extras
 Several common environment Wrappers are also kept [here](https://github.com/NickLucche/avalanche/blob/master/avalanche/training/strategies/reinforcement_learning/utils.py) as we encourage the use of this pattern to suit environments output to your needs. 
 We also provide common [gym control environments](https://github.com/NickLucche/avalanche/blob/master/avalanche/envs/classic_control.py) which have been "parametrized" so you can tweak values such as force and gravity to help out in testing new ideas in a fast and reliable way on well known testbeds. These environments are available by pre-pending a `C` to the env id as in `CCartPole-v1` as they're registered on first import.
 
 ### Models
 In this module you can find an implementation of both MLPs and CNNs for deep-q learning and actor-critic approaches, adapted from popular papers such as "Human-level Control Through Deep Reinforcement Learning" and "Overcoming catastrophic forgetting in neural networks" to learn directly from pixels or states.  
 
 ### Logging
 A Tqdm-based interactive logger has been added to ease readability as well as sensible default loggers for RL algorithms.



## Quick Example
----------------

```python
import torch
from torch.optim import Adam
from avalanche.benchmarks.generators.rl_benchmark_generators import gym_benchmark_generator

from avalanche.models.actor_critic import ActorCriticMLP
from avalanche.training.strategies.reinforcement_learning import A2CStrategy

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Model
model = ActorCriticMLP(num_inputs=4, num_actions=2, actor_hidden_sizes=1024, critic_hidden_sizes=1024)

# CRL Benchmark Creation
scenario = gym_benchmark_generator(['CartPole-v1'], n_experiences=1, n_parallel_envs=1, 
    eval_envs=['CartPole-v1'])

# Prepare for training & testing
optimizer = Adam(model.parameters(), lr=1e-4)

# Reinforcement Learning strategy
strategy = A2CStrategy(model, optimizer, per_experience_steps=10000, max_steps_per_rollout=5, 
    device=device, eval_every=1000, eval_episodes=10)

# train and test loop
results = []
for experience in scenario.train_stream:
    strategy.train(experience)
    results.append(strategy.eval(scenario.test_stream))
```
Compare it with [vanilla Avalanche snippet](https://avalanche.continualai.org/)!

Check out more examples [here](https://github.com/NickLucche/avalanche/blob/master/examples/reinforcement_learning/) (advanced ones coming soon) or in unit tests. We also got a small-scale reproduction of the original EWC paper (Deepmind) [experiments](https://github.com/NickLucche/avalanche/blob/master/examples/reinforcement_learning/ewc.py).

## Installation
______________
`pip install git+https://github.com/ContinualAI/avalanche-rl.git` for installing the latest available version from the master branch.
Alternatively, you can build the Dockerfile in this repo yourself or run the one on DockerHub with `docker run -it --rm nicklucche/avalanche-rl:latest bash`. 

You can setup Avalanche RL in "dev-mode" by simply cloning this repo and install it using pip:  
```
    git clone https://github.com/ContinualAI/avalanche-rl
    cd avalanche-rl
    pip install -r requirements.txt
    pip install -e .
``` 

Be aware that Atari Roms are not included in the installation, please refer to https://github.com/openai/atari-py#roms for details. 

Disclaimer
----------------
This project is under strict development so expect changes on the main branch on a fairly regular basis. As Avalanche itself it's still in its early Alpha versions, it's only fair to say that Avalanche RL is in super-duper Alpha.

We believe there's lots of room for improvements and tweaking but at the same time there's much that can be offered to the growing community of continual learning practitioners approaching reinforcement learning by allowing to perform experiments under a common framework with a well-defined structure.

Cite Avalanche RL
----------------
If you used Avalanche RL in your research project, please remember to cite our reference paper published at [ICIAP2021](https://www.iciap2021.org/): ["Avalanche RL: a Continual Reinforcement Learning Library"](https://arxiv.org/abs/2202.13657). 

```
@misc{https://doi.org/10.48550/arxiv.2202.13657,
  doi = {10.48550/ARXIV.2202.13657},
  url = {https://arxiv.org/abs/2202.13657},
  author = {Lucchesi, Nicolò and Carta, Antonio and Lomonaco, Vincenzo and Bacciu, Davide},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Avalanche RL: a Continual Reinforcement Learning Library},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

