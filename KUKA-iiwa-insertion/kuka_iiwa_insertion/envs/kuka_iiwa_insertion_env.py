import pybullet as p
import gym
import matplotlib.pyplot as plt
import numpy as np
import time
from numpy import sin, cos

from ..robot.kuka_iiwa import KukaIIWA


class IiwaInsertionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(IiwaInsertionEnv, self).__init__()
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1]*3),
            high=np.array([1]*3))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1]*3),
            high=np.array([1]*3))
        self.np_random, _ = gym.utils.seeding.np_random()
        self.client = p.connect(p.GUI)  # DIRECT
        self.closed = False
        self.rendered_img = None
        self.reset()

    def reset(self):
        p.resetSimulation(self.client)
        self.kuka_iiwa = KukaIIWA(self.client)

        self._generate_target()
        self.action_step_size =  np.random.uniform(0.0005,0.0015)

        return self.kuka_iiwa.get_observation()
    
    def _generate_target(self):
        while True:
            distance = np.random.uniform(0.4,0.7)
            angle = np.random.uniform(-np.pi/2, np.pi/2)
            height = np.random.uniform(0.2,0.6)
            

            position_canidate = [distance * np.cos(angle), distance * np.sin(angle), height]
            if self.kuka_iiwa.inverse_kinematics(position_canidate, [np.pi, 0, np.pi]):
                break

        self.target_position = position_canidate

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
        action = [self.action_step_size *  a for a in action]
        self.kuka_iiwa.apply_action(action)
        p.stepSimulation()
        observation = self.kuka_iiwa.get_observation()[:3]
        reward = self.calculate_reward(observation)

        return self.target_position[:3]-np.array(observation[:3]), reward, reward > -0.05, {}

    def calculate_reward(self, observation):
        return -np.linalg.norm(self.target_position[:3]-np.array(observation[:3]))

    def __del__(self):
        if not self.closed:
            self.close()
