import gym
import numpy as np

import kuka_iiwa_insertion

from stable_baselines3 import SAC


def play():
    model = SAC.load("models/kuka_iiwa_insertion-v0_sac")

    env = gym.make('kuka_iiwa_insertion-v0', use_gui=True)

    obs = env.reset()
    i = 0
    while True:
        i += 1
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if i % 100 == 0 or dones: 
            print(obs, rewards, dones, info)
        if dones:
            print("="*20 + " RESET " + "="*20)
            env.reset()
        #env.render()

if __name__ == "__main__":
    play()