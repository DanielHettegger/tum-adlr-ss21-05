import gym
import numpy as np
import kuka_iiwa_insertion
import time


env = gym.make('kuka_iiwa_insertion-v0', use_gui=True)
#env.reset()
observation = [0,0,0]
current_task_id = 0
for i in range(100000):
    #env.render()
    action = [0.5*np.sin(i/100),0.5*np.cos(i/100),0]
         
    observation, reward, done, _ = env.step(action)#env.action_space.sample()) # take a random action
    if i % 1000 == 0:
        print(observation, reward, done)

    time.sleep(1.0/240.0)

env.close()