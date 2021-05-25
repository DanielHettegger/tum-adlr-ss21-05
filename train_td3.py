import gym
import numpy as np

import kuka_iiwa_insertion

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make('kuka_iiwa_insertion-v0', use_gui=False)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
try:
    model = TD3.load("kuka_iiwa_insertion-v0",env, action_noise=action_noise, verbose=1)
except:
    model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1)

i = 0
save_interval = 10000
while True:
    i += save_interval
    model.learn(total_timesteps=save_interval, log_interval=10)
    model.save("kuka_iiwa_insertion-v0")
