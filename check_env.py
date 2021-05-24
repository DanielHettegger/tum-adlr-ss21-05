from stable_baselines3.common.env_checker import check_env
import kuka_iiwa_insertion
import gym

env = gym.make('kuka_iiwa_insertion-v0', use_gui=False)
# It will check your custom environment and output additional warnings if needed
check_env(env)