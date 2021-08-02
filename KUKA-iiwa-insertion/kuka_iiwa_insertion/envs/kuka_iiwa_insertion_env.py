import pybullet as p
import pybullet_data
import gym
import matplotlib.pyplot as plt
import numpy as np
import time
from numpy import sin, cos

from ..files import get_resource_path
from ..robot.kuka_iiwa import KukaIIWA
from .disturbance_force import StaticForce, SpringForceXY, DisturbanceForce


class IiwaInsertionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,  steps_per_action=100, max_steps=100, action_step_size=0.005, tasks=None, use_gui=False, apply_disturbance_force=True):
        super(IiwaInsertionEnv, self).__init__()
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1]*3),
            high=np.array([1]*3))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1]*6),
            high=np.array([1]*6))

        self.max_observation = 2.0
        self.target_size = 0.05

        self.base_position = [0.6, 0.0, 0.0]
        self.kuka_reset_position = [0.6, 0.0, 0.4]

        x_limits = [self.kuka_reset_position[0]-0.1, self.kuka_reset_position[0]+0.1]
        y_limits = [self.kuka_reset_position[1]-0.1, self.kuka_reset_position[1]+0.1]
        z_limits = [0, self.kuka_reset_position[2]+0.1]
        self.limits = [x_limits, y_limits, z_limits]

        self.action_step_size = action_step_size

        self.max_steps = max_steps
        self.steps_per_action = steps_per_action
        self.apply_disturbance_force = apply_disturbance_force

        self.observation_position = None
        self.last_observation_position = None

        self.np_random, _ = gym.utils.seeding.np_random()

        if tasks is None:
            self.tasks = [
                            ("square", "none"),
                            ("square", "static"),
                            ("square", "spring"),
                            #("zylindric", "none"),
                            #("zylindric", "static"),
                            #("zylindric", "spring"),
                         ]
        else:
            self.tasks = tasks
        
        self.number_of_tasks = len(self.tasks)
        self.current_task = 0

        self.use_gui = use_gui
        if use_gui:
            self.client = p.connect(p.GUI) 
        else:
            self.client = p.connect(p.DIRECT)
        p.setGravity(0,0,0) #disable gravity for simulated gravity compensation
        self.closed = False
        self.visual_target = None
        self.kuka_iiwa = KukaIIWA(self.client, self.kuka_reset_position, tool=self.tasks[self.current_task][0], control_mode="impedance")
        self._setup_task()
        self.rendered_img = None
        self.reset()
    
    def _generate_task(self):
        self._generate_force_disturbance()
        self._generate_target_position()

    def _generate_target_position(self):
        self.target_position = np.array([0.6,0.0,0.05])
        self._update_visual_target()
    
    def _generate_force_disturbance(self, disturbance="none"):
        self.disturbance.generate()

    def _setup_task(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_plane_id = p.loadURDF("plane.urdf")
        self.base_id = p.loadURDF(get_resource_path('kuka_iiwa_insertion','models', 'square10x10', 'base10x10.urdf'),
                    basePosition=self.base_position,
                    physicsClientId=self.client)
        self._setup_disturbance(self.tasks[self.current_task][1])

    def _setup_disturbance(self, disturbance="none"):
        if isinstance(disturbance, str): 
            if disturbance is "static":
                self.disturbance = StaticForce()
            elif disturbance is "spring":
                self.disturbance = SpringForceXY()
            else:
                self.disturbance = StaticForce(magnitude=0)
        elif isinstance(disturbance, DisturbanceForce):
            self.disturbance = disturbance
        else:
            self.disturbance = StaticForce(magnitude=0)
     
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
        self.kuka_iiwa.reset() 
        self._generate_task()
        self.steps = 0

        return np.append(self.get_observation(),[0,0,0])

    def reset_task(self, task_id):
        print("Resetting to task {}".format(task_id))
        if self.current_task is not task_id:
            tool, disturbance = self.tasks[task_id]
            if self.kuka_iiwa.tool is not tool:
                self.kuka_iiwa.reset_tool(tool)
            self._setup_disturbance(disturbance)
            self.current_task = task_id
        return self.reset()

    def render(self):
        # Base information
        kuka_id, client_id = self.kuka_iiwa.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(kuka_id, client_id)]

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = [0,0,1]
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        _,_,frame,_,_ = p.getCameraImage(800, 800)#, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (800, 800, 4))
        self.rendered_img.set_data(frame)
        return frame

    def close(self): 
        self.closed = True
        p.disconnect(self.client)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.steps += 1
        scaled_action = (self.action_step_size * np.array(action))

        position = np.copy(self.observation_position)
        for i, limit in enumerate(self.limits):
            position[i] = np.clip(position[i], limit[0], limit[1])

        self.kuka_iiwa.apply_action(scaled_action, position)
        for i in range (self.steps_per_action):
            self.kuka_iiwa.step_controller()
            self.kuka_iiwa.apply_tcp_force(self.disturbance.get_force(self.observation_position))
            p.stepSimulation()
        observation = self.get_observation()
        reward = self.calculate_reward(observation, action)

        observation_with_velocity = np.append(
                    observation / self.max_observation,
                    (self.observation_position-self.last_observation_position) / self.action_step_size
                )

        return observation_with_velocity,  reward, self.is_done(observation), {}

    def calculate_reward(self, observation, action):
        reward = 0
        if self.last_observation_position is not None:
            reward += ( #np.linalg.norm(self.target_position - self.last_observation_position) - np.linalg.norm(observation) # reward for incremental improvements
                      -np.linalg.norm(observation)
                      -2.0 * np.linalg.norm(observation[:2]) # punishment for xy deviation
                      -0.002) # punishment for every timestep
        if np.abs(action[2]) < 0.6:
            reward -= 1 - np.abs(action[2]) # punish small z actions
        if (np.linalg.norm(observation) < self.target_size).item():
            reward += 3 # reward for finishing 
        return reward

    def get_observation(self):
        current_position = np.array(self.kuka_iiwa.get_observation()[:3])
        if self.observation_position is not None:
            self.last_observation_position = self.observation_position
        self.observation_position = current_position
        return self.target_position - current_position
    
    def is_done(self,observation):
        return (np.linalg.norm(observation) < self.target_size).item() or self.steps > self.max_steps

    def __del__(self):
        if not self.closed:
            self.close()
