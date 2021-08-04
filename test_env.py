import gym
import numpy as np
import kuka_iiwa_insertion
import time


env = gym.make('kuka_iiwa_insertion-v0', use_gui=True, max_steps=100, sleep_on_timestep=(1./2000.))
#env.reset()
observation = [0,0,0]
current_task_id = 0
rewards  = []
steps = 0
for i in range(100000):
    #env.render()
    action = observation[:3]
    max_action = np.max(np.abs(action))
    if max_action > 0.01:
        action = [a / max_action for a in action] 

    observation, reward, done, _ = env.step(action)#env.action_space.sample()) # take a random action
    rewards += [reward]
    steps+=1



    if done:
        print("Sum of rewards {}, steps {}".format(np.sum(rewards),steps))
        rewards  = []
        steps = 0
        current_task_id += 1
        current_task_id %= env.number_of_tasks
        print("Resetting Env to id {}".format(current_task_id))
        env.reset_task(current_task_id)
        time.sleep(0.5)

env.close()