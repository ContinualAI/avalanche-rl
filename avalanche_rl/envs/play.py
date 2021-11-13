import argparse
import gym
from avalanche_rl.envs import *
import cv2 
from argparse import ArgumentParser
import argparse

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('-e', '--env', help='Environment id',
                      default='CCartPole-v1')
    # any other argument to pass to the env
    # args.add_argument('args', nargs=argparse.REMAINDER)
    args, extras = args.parse_known_args()

    # env = gym.make(args.env, gravity=0.001, length=1., masscart=1000., force_mag=1.)
    extras_dict = {}
    for i in range(0, len(extras), 2):
        key, val = extras[i:i+2]
        key = key.replace('-', '')
        if val.isdigit():
            val = int(val)
        elif val.replace('.', '').isdigit():
            val = float(val)
        elif val.lower() == 'true' or val.lower() == 'false':
            val = bool(val)

        extras_dict[key] = val

    print('extra arguments for creating environment:', extras_dict)
    env = gym.make(args.env, **extras_dict)

    cv2.namedWindow('window')
    for _ in range(3):
        done = False
        env.reset()
        action = None
        while not done:
            key = cv2.waitKey(25)
            if key == ord('q'):
                exit()
            for i in range(10):
                if key == ord(str(i)):
                    action = i
            if action is not None:
                obs, _, done, _ = env.step(action)
            obs = env.render('rgb_array')
            # print(obs.shape)
    cv2.destroyAllWindows()
