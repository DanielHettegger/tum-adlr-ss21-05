import gym
import kuka_iiwa_insertion

env = gym.make('kuka_iiwa_insertion-v0')
#env.reset()
for i in range(100000):
    #env.render()
    update = env.step(env.action_space.sample()) # take a random action
    if i % 1000 == 0:
        print(update)
env.close()