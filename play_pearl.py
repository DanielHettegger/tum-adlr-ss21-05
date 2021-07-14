import gym
import numpy as np

import kuka_iiwa_insertion

from stable_baselines3 import PEARL


def play():
    env = gym.make('kuka_iiwa_insertion-v0', use_gui=True)
    model = PEARL.load("models/kuka_iiwa_insertion-v0_pearl_best_model", env=env)

    obs = env.reset()
    i = 0
    episode_reward = 0.0
    while True:
        i += 1
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        episode_reward += rewards
        if i % 10 == 0 or dones: 
            print(obs, episode_reward, rewards, info)
        if dones:
            print("="*20 + " RESET " + "="*20)
            episode_reward = 0
            obs = env.reset_task(np.random.randint(0,3))

        #env.render()

if __name__ == "__main__":
    play()