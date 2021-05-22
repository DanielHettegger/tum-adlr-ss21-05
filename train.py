import os

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import mSAC
from stable_baselines3.common.evaluation import evaluate_policy, evaluate_meta_policy

from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecCheckNan,
    DummyVecEnv,
    VecNormalize,
)

import random
import datetime
import time
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

import numpy as np
import torch
import gym

import kuka_iiwa_insertion

#from agent import TRPOAgent


def main():
    env = gym.make('kuka_iiwa_insertion-v0')
    env.reset()
    for _ in range(100000):
        # env.render()
        obs, rewards, dones, info = env.step(
            env.action_space.sample())  # take a random action
        print('Action: ', env.action_space.sample())
        print('observation: ', obs[:3])
        print('observation size: ', env.observation_space)
    env.close()


if __name__ == "__main__":
    main()
