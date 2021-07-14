import gym
import numpy as np

import kuka_iiwa_insertion

from stable_baselines3 import SAC


def play():
    env = gym.make('kuka_iiwa_insertion-v0', use_gui=True)
    model = SAC.load("models/kuka_iiwa_insertion-v0_sac_best_model", env=env)


    obs = env.reset()
    i = 0
    episode_reward = 0.0
    while True:
        i += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_reward += rewards
        if i % 10 == 0 or dones: 
            print(obs, episode_reward, rewards, info)
        if dones:
            print("="*20 + " RESET " + "="*20)
            episode_reward = 0
            env.reset()
        #env.render()

if __name__ == "__main__":
    play()