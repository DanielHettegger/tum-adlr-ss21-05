import pybullet as p
import gym
import matplotlib.pyplot as plt
import numpy as np
import time
from numpy import sin, cos

from ..robot.kuka_iiwa import KukaIIWA


class IiwaInsertionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps=1000, use_gui=False):
        super(IiwaInsertionEnv, self).__init__()
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1]*3),
            high=np.array([1]*3))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1]*3),
            high=np.array([1]*3))

        self.max_observation = 2.0
        self.target_size = 0.05

        self.max_steps = max_steps

        self.np_random, _ = gym.utils.seeding.np_random()

        self.use_gui = use_gui
        if use_gui:
            self.client = p.connect(p.GUI) 
        else:
            self.client = p.connect(p.DIRECT)
        self.closed = False
        self.visual_target = None
        self.kuka_iiwa = KukaIIWA(self.client)
        self.rendered_img = None
        self.reset()
    
    def _generate_target_position(self):
        while True:
            distance = np.random.uniform(0.4,0.7)
            angle = np.random.uniform(-np.pi/2, np.pi/2)
            height = np.random.uniform(0.2,0.6)
            
            position_canidate = [distance * np.cos(angle), distance * np.sin(angle), height]
            if self.kuka_iiwa.inverse_kinematics(position_canidate, [np.pi, 0, np.pi]):
                break

        self.target_position = np.array(position_canidate)
        self._update_visual_target()

    def _update_visual_target(self):
         if self.use_gui:
            if self.visual_target is None:
                vuid = p.createVisualShape(p.GEOM_SPHERE, 
                    radius=self.target_size, 
                    physicsClientId=self.client,
                    rgbaColor=[1, 0, 0, 0.5],)
                self.visual_target = p.createMultiBody(baseMass=0,
                    baseVisualShapeIndex=vuid, 
                    basePosition=self.target_position, 
                    physicsClientId=self.client)
            else:
                p.resetBasePositionAndOrientation(self.visual_target,
                    posObj=self.target_position,
                    ornObj=[0,0,0,1],
                    physicsClientId=self.client)

    def reset(self):
        #p.resetSimulation(self.client)
        self.kuka_iiwa.reset()
        
        self._generate_target_position()
        self.action_step_size =  np.random.uniform(0.0005,0.0015)
        self.steps = 0

        return self.get_observation()

    def render(self):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((800, 800, 4)))

        # Base information
        kuka_id, client_id = self.kuka_iiwa.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(kuka_id, client_id)]

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = [0,0,1]#np.matmul(rot_mat, np.array([1, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        _,_,frame,_,_ = p.getCameraImage(800, 800)#, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (800, 800, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)
        return frame

    def close(self): 
        self.closed = True
        p.disconnect(self.client)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.steps += 1
        action = [self.action_step_size *  a for a in action]
        self.kuka_iiwa.apply_action(action)
        p.stepSimulation()
        observation = self.get_observation()
        reward = self.calculate_reward(observation)

        return observation / self.max_observation, reward, self.is_done(observation), {}

    def calculate_reward(self, observation):
        return -np.linalg.norm(observation)

    def get_observation(self):
        observation = np.array(self.kuka_iiwa.get_observation()[:3])
        return self.target_position - observation
    
    def is_done(self,observation):
        return (np.linalg.norm(observation) < self.target_size).item() or self.steps > self.max_steps

    def __del__(self):
        if not self.closed:
            self.close()
