import gym
import numpy as np

import kuka_iiwa_insertion

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def play():
    model = TD3.load("kuka_iiwa_insertion-v0")

    env = gym.make('kuka_iiwa_insertion-v0', use_gui=True)

    obs = env.reset()
    i = 0
    while True:
        i += 1
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if i % 100 == 0: 
            print(obs, rewards, dones, info)
        if dones:
            print("="*20 + " reset " + "="*20)
            env.reset()
        #env.render()

if __name__ == "__main__":
    play()