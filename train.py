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

import KUKA-iiwa-insertion

if __name__ == "__main__":
    main()

from agent import TRPOAgent

def main():
    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                             torch.nn.Linear(64, 2))
    agent = TRPOAgent(policy=nn)

    #agent.load_model("agent.pth")
    agent.train("kuka_iiwa_insertion-v0", seed=0, batch_size=5000, iterations=100,
                max_episode_length=250, verbose=True)
    agent.save_model("agent.pth")

    env = gym.make('kuka_iiwa_insertion-v0')
    ob = env.reset()
    iterator = 1
    while True:
        action = agent(ob)
        ob, _, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1./50.)
