import pybullet as p
import gym
import numpy as np
import time
from numpy import sin, cos

from ..robot.kuka_iiwa import KukaIIWA

class IiwaInsertionEnv(gym.env):
    metadata = {'render.modes': ['human']}  
  
    def __init__(self):
        super(IiwaInsertionEnv, self).__init__()
        self.action_space = gym.spaces.box.Box(
            low= np.array([-0.01]*3),
            high=np.array([ 0.01]*3))
        self.observation_space = gym.spaces.box.Box(
            low= np.array([-10]*3),
            high=np.array([ 10]*3))
        self.np_random, _ = gym.utils.seeding.np_random()
        self.client = p.connect(p.DIRECT)
        self.target = np.array([1,1,1])
        self.reset()

    def reset(self):
        self.physicsClient = p.connect(p.GUI)
        self.kuka_iiwa = KukaIIWA()
        return self.kuka_iiwa.get_observation()

    def render(self):
        p.getCameraImage()

    def close(self):
        p.disconnect(self.client)    
    
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.kuka_iiwa.applyAction(action)
        p.stepSimulation(1./50.)
        observation = self.kuka_iiwa.get_observation()
        reward = self.calculate_reward(observation)

        return observation, reward, self.reward > 100, {}

    def calculate_reward(self, observation):
        return 1/(np.norm(self.target-np.array(observation))+0.0001)


    def __del__(self):
        self.close()

