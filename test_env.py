import gym
import kuka_iiwa_insertion

env = gym.make('kuka_iiwa_insertion-v0')
env.reset()
for _ in range(100000):
    #env.render()
    env.step([0,0,0,0]) # take a random action
env.close()