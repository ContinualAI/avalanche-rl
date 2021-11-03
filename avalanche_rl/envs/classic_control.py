from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.classic_control.mountain_car import MountainCarEnv
from gym.envs.classic_control.acrobot import AcrobotEnv
import math
import numpy as np
from gym import spaces


class ContinualCartPoleEnv(CartPoleEnv):

    def __init__(
            self, gravity=9.8, masscart=1.0, masspole=0.1, length=0.5,
            force_mag=10.0, tau=0.02, theta_threshold_radians=12 * 2 * math.pi /
            360, x_threshold=2.4, seed: int = None):
        super().__init__()
        self.gravity = gravity 
        self.masscart = masscart 
        self.masspole = masspole
        self.length = length
        self.force_mag = force_mag 
        self.tau = tau 
        self.theta_threshold_radians = theta_threshold_radians
        self.x_threshold = x_threshold

        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)


class ContinualMountainCarEnv(MountainCarEnv):

    def __init__(self, 
                goal_velocity=0., 
                min_position=-1.2,
                max_position=0.6,
                max_speed=0.07,
                goal_position=0.5,
                force=0.001,
                gravity=0.0025, 
                seed=None):
        super().__init__(goal_velocity=goal_velocity)
        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.goal_position = goal_position

        self.force = force
        self.gravity = gravity

        self.low = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )
        self.seed(seed)


class ContinualAcrobotEnv(AcrobotEnv):

    def __init__(self, 
                 link_length_1=1.,
                 link_length_2=1.,
                 link_mass_1=1.,
                 link_mass_2=1.,
                 link_com_pos_1=0.5,
                 link_com_pos_2=0.5,
                 link_moi=1.,
                 max_vel_1=4 * math.pi,
                 max_vel_2=9 * math.pi,
                 avail_torque=[-1., 0., +1],
                 torque_noise_max=0.,
                 seed=None):
        super().__init__()
        self.LINK_LENGTH_1 = link_length_1
        self.LINK_LENGTH_2 = link_length_2
        self.LINK_MASS_1 = link_mass_1
        self.LINK_MASS_2 = link_mass_2
        self.LINK_COM_POS_1 = link_com_pos_1
        self.LINK_COM_POS_2 = link_com_pos_2
        self.LINK_MOI = link_moi
        self.MAX_VEL_1 = max_vel_1 
        self.MAX_VEL_2 = max_vel_2
        self.AVAIL_TORQUE = avail_torque
        self.torque_noise_max = torque_noise_max

        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2],
            dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)
        self.seed(seed)
