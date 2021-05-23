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
            low=np.array([-0.01]*3),
            high=np.array([0.01]*3))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-10]*6),
            high=np.array([10]*6))
        self.np_random, _ = gym.utils.seeding.np_random()
        self.client = p.connect(p.GUI)  # DIRECT
        self.closed = False
        self.rendered_img = None
        self.reset()

    def reset(self):
        p.resetSimulation(self.client)
        self.kuka_iiwa = KukaIIWA(self.client)

        self.target = np.random.uniform(0.2,0.4,3)

        return self.kuka_iiwa.get_observation()

    def render(self):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((800, 800, 4)))

        # Base information
        kuka_id, client_id = self.kuka_iiwa.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(kuka_id, client_id)]
        pos[2] = 0.5
        pos[0] = -0.7

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([1, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(800, 800, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (800, 800, 4))
        return frame

    def close(self):
        self.closed = True
        p.disconnect(self.client)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.kuka_iiwa.apply_action(action)
        p.stepSimulation()
        observation = self.kuka_iiwa.get_observation()
        reward = self.calculate_reward(observation)

        return observation, reward, reward < 0.05, {}

    def calculate_reward(self, observation):
        return -np.linalg.norm(self.target[:3]-np.array(observation[:3]))

    def __del__(self):
        if not self.closed:
            self.close()
