import gym
import numpy as np
import kuka_iiwa_insertion


env = gym.make('kuka_iiwa_insertion-v0', use_gui=True, max_steps=100)
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

    #action = np.sign(observation[:3])
    #print(action) 
    observation, reward, done, _ = env.step(action)#env.action_space.sample()) # take a random action
    rewards += [reward]
    steps+=1
    #if i % 100 == 0:
    #    print(observation, reward, done)


    if done:
        print("Sum of rewards {}, steps {}".format(np.sum(rewards),steps))
        rewards  = []
        steps = 0
        env.reset()
    
    if i % 300 == 0 and i is not 0:
        current_task_id += 1
        current_task_id %= env.number_of_tasks
        steps = 0
        print("Resetting Env to id {}".format(current_task_id))
        env.reset_task(current_task_id)

env.close()