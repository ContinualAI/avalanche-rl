from gym.envs.registration import register

__version__ = "0.0.1"

# Register continual version of gym envs
register(
    id="CCartPole-v1",
    entry_point="avalanche_rl.envs.classic_control:ContinualCartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="CMountainCar-v0",
    entry_point="avalanche_rl.envs.classic_control:ContinualMountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="CAcrobot-v1",
    entry_point="avalanche_rl.envs.classic_control:ContinualAcrobotEnv",
    reward_threshold=-100.0,
    max_episode_steps=500,
)
