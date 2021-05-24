import gym
import numpy as np
import kuka_iiwa_insertion


env = gym.make('kuka_iiwa_insertion-v0', use_gui=True)
#env.reset()
observation = [0,0,0]
for i in range(100000):
    #env.render()
    action = observation
    max_action = np.max(np.abs(action))
    if max_action > 0.01:
        action = [a / max_action for a in action] 
         
    observation, reward, done, _ = env.step(action)#env.action_space.sample()) # take a random action
    if i % 1000 == 0:
        print(observation, reward, done)

    if done:
        env.reset()
env.close()