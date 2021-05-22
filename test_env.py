import gym
import kuka_iiwa_insertion

env = gym.make('kuka_iiwa_insertion-v0')
env.reset()
for _ in range(100000):
    #env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()