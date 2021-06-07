import gym
import numpy as np

import kuka_iiwa_insertion


from stable_baselines3 import SAC

env = gym.make('kuka_iiwa_insertion-v0', use_gui=False)

try:
    model = SAC.load("models/kuka_iiwa_insertion-v0_sac",env, verbose=1)
except:
    model = SAC("MlpPolicy", env, verbose=1)

i = 0
save_interval = 10000
while True:
    i += save_interval
    model.learn(total_timesteps=save_interval, log_interval=10)
    model.save("models/kuka_iiwa_insertion-v0_sac")
