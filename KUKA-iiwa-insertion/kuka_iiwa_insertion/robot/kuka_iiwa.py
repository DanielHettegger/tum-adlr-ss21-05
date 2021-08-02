#taken and modified from:
#https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kuka.py
import os, inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(os.path.dirname(currentdir))
#os.sys.path.insert(0, parentdir)

from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed
import pybullet as p
import pybullet_data


import numpy as np
from numpy import pi, sqrt, cos, sin, arctan2, array, matrix
from numpy.linalg import norm
#import copy
import math
from ..files import get_resource_path

from scipy.spatial.transform import Rotation as R


class KukaIIWA:
  def __init__(self,client, reset_position = [0.6, 0, 0.4], tool = None, control_mode=None):
    self.client = client

    self.ee_index = 8

    self.l02 = 0.36
    self.l24 = 0.42
    self.l46 = 0.4
    self.l6E = 0.126 #+ 0.026

    self.robot_name = 'iiwa'

    self.max_force = 800
    self.max_velocity = 0.1

    self.reset_position = reset_position

    self.joint_names = ['{}_joint_1'.format(self.robot_name),
                        '{}_joint_2'.format(self.robot_name),
                        '{}_joint_3'.format(self.robot_name),
                        '{}_joint_4'.format(self.robot_name),
                        '{}_joint_5'.format(self.robot_name),
                        '{}_joint_6'.format(self.robot_name),
                        '{}_joint_7'.format(self.robot_name)]

    self.joint_limits = {'{}_joint_1'.format(self.robot_name):{'lower': np.deg2rad(-170.0), 'upper':np.deg2rad(170.0)},
                         '{}_joint_2'.format(self.robot_name):{'lower': np.deg2rad(-120.0), 'upper':np.deg2rad(120.0)},
                         '{}_joint_3'.format(self.robot_name):{'lower': np.deg2rad(-170.0), 'upper':np.deg2rad(170.0)},
                         '{}_joint_4'.format(self.robot_name):{'lower': np.deg2rad(-120.0), 'upper':np.deg2rad(120.0)},
                         '{}_joint_5'.format(self.robot_name):{'lower': np.deg2rad(-170.0), 'upper':np.deg2rad(170.0)},
                         '{}_joint_6'.format(self.robot_name):{'lower': np.deg2rad(-120.0), 'upper':np.deg2rad(120.0)},
                         '{}_joint_7'.format(self.robot_name):{'lower': np.deg2rad(-175.0), 'upper':np.deg2rad(175.0)},
                         }
    self.joint_indices = [i+1 for i in range(len(self.joint_names))]
    
    self.tool = tool
    if tool is None:
      self.kuka_uid = p.loadURDF(get_resource_path('kuka_iiwa_insertion','robot', 'urdf', 'iiwa14.urdf'), physicsClientId=self.client)
    else:
      self._load_with_tool(tool)

    if control_mode == None:
      self.control_mode = ["position"]
    elif control_mode in ["position", "impedance"]:
      self.control_mode = control_mode
    else:
      print("kuka_iiwa.py __init__ Unknown control type. Defaultig to position control.")
      self.control_mode = ["position"]

    #self.maxForceSlider = p.addUserDebugParameter("maxForce", 0, 1600, 800)
    self.reset()

  def _load_with_tool(self, tool):
    
    p0 = bc.BulletClient(connection_mode=p.DIRECT)
    p1 = bc.BulletClient(connection_mode=p.DIRECT)

    kuka = p0.loadURDF(get_resource_path('kuka_iiwa_insertion','robot', 'urdf', 'iiwa14.urdf'))
    if tool is "square":
      tool = p1.loadURDF(get_resource_path('kuka_iiwa_insertion','models', 'square10x10', 'tool9x9.urdf'))
    elif tool is "triangular":
      tool = p1.loadURDF(get_resource_path('kuka_iiwa_insertion','models', 'triangle10x10', 'TriangleTool9x9.urdf'))
    elif tool is "zylindric":
      tool = p1.loadURDF(get_resource_path('kuka_iiwa_insertion','models', 'zylinder10x10', 'ZylinderTool9x9.urdf'))
    else:
      raise NotImplementedError("This tool has not been implemented")

    ed0 = ed.UrdfEditor()
    ed0.initializeFromBulletBody(kuka, p0._client)
    ed1 = ed.UrdfEditor()
    ed1.initializeFromBulletBody(tool, p1._client)

    jointPivotXYZInParent = [0, 0, 0.026]
    jointPivotRPYInParent = [0, 0, 0]

    jointPivotXYZInChild = [0, 0, 0]
    jointPivotRPYInChild = [0, 0, 0]

    newjoint = ed0.joinUrdf(ed1, self.ee_index, jointPivotXYZInParent, jointPivotRPYInParent,
                            jointPivotXYZInChild, jointPivotRPYInChild, p0._client, p1._client)

    newjoint.joint_type = p.JOINT_FIXED
    
    self.kuka_uid = ed0.createMultiBody([0, 0, 0], [0,0,0,1], self.client)

  def reset_tool(self, tool):
     p.removeBody(self.kuka_uid)
     self._load_with_tool(tool)
     self.reset()

  def reset(self):
    self.rs = 2
    self.tr = 0.0

    self.target_position =    self.reset_position
    self.target_orientation = [pi, 0, pi]

    
    q = self.inverse_kinematics(self.target_position, self.target_orientation)
    self.target_q = q
    if q:
      self.force_joint_targets(q)

    # disable position controller 
    p.setJointMotorControlArray(bodyUniqueId= self.kuka_uid,
                              jointIndices= self.joint_indices,
                              controlMode= p.POSITION_CONTROL,
                              targetPositions= [0] * len(self.joint_names),
                              targetVelocities= [0] * len(self.joint_names),
                              forces= [0] * len(self.joint_names),
                              physicsClientId = self.client)
  
  def get_ids(self):
    return self.client, self.kuka_uid

  def get_action_dimension(self):
    return 3

  def get_observation_dimension(self):
    return len(self.get_observation())

  def get_observation(self):
    observation = []
    state = p.getLinkState(self.kuka_uid, self.ee_index)
    pos = state[0]
    orn = state[1]
    euler = p.getEulerFromQuaternion(orn)

    observation.extend(list(pos))
    observation.extend(list(euler))

    return observation

  def apply_action(self, action, position = None):
    if position is None:
      target_candidate = self.target_position
    else:
      target_candidate = position
              
    target_candidate[0] += action[0]
    target_candidate[1] += action[1]
    target_candidate[2] += action[2]
    #self.target_orientation[2] += action[3]
    
    q = self.inverse_kinematics(target_candidate, self.target_orientation) 
    if q:
      self.target_q = q
      self.target_position = target_candidate
      #print("Target position: {}\nActions: {}".format(target_candidate, action))

  def step_controller(self):
    if self.control_mode is "position":
      p.setJointMotorControlArray(bodyUniqueId= self.kuka_uid,
                            jointIndices= self.joint_indices,
                            controlMode= p.POSITION_CONTROL,
                            targetPositions= self.target_q,
                            targetVelocities= [0] * len(self.joint_names),
                            forces= [self.max_force] * len(self.joint_names),
                            physicsClientId = self.client)

    elif self.control_mode is "impedance":
      states = p.getJointStates(bodyUniqueId = self.kuka_uid,
                                  jointIndices = self.joint_indices,
                                  physicsClientId = self.client)

      q = [s[0] for s in states]
      q_dot = [s[1] for s in states]

      M = p.calculateMassMatrix(bodyUniqueId= self.kuka_uid, objPositions = q, physicsClientId = self.client)

      q = np.array(q)
      q_des = np.array(self.target_q)
      q_dot = np.array(q_dot)
      M = np.array(M)
      K = np.eye(len(q)) * 90.0
      D = np.eye(len(q)) * 0.2

      t = M @ (-K @ (q - q_des) - D @ q_dot)
      
      p.setJointMotorControlArray(bodyUniqueId= self.kuka_uid,
                            jointIndices= self.joint_indices,
                            controlMode= p.TORQUE_CONTROL,
                            targetPositions= [0] * len(self.joint_names),
                            targetVelocities= [0] * len(self.joint_names),
                            forces= t,
                            physicsClientId = self.client)
    
  
  def force_joint_targets(self, q):
    for i, qi in enumerate(q):
      p.resetJointState(bodyUniqueId= self.kuka_uid,
                        jointIndex= i+1,
                        targetValue= qi,
                        physicsClientId= self.client)
  
  def apply_tcp_force(self, force):
    p.applyExternalForce(self.kuka_uid,
                         self.ee_index-2,
                         force,
                         [0,0,0],
                         p.WORLD_FRAME,
                         self.client)


  def inverse_kinematics(self, position, orientation, redundancy=None, redundancy_status=None):
    if redundancy_status is not None:
      self.rs = redundancy_status

    if redundancy is not None:
      self.tr = redundancy

    q = 7 * [0.0]
    pE0 = matrix([[p] for p in position])
    RE0 = rotation_matrix(orientation[0], orientation[1], orientation[2])

    rs2 = - np.sign((self.rs & (1 << 0))-0.5)
    rs4 = - np.sign((self.rs & (1 << 1))-0.5)
    rs6 = - np.sign((self.rs & (1 << 2))-0.5)

    pE6 = matrix([[0.0], [0.0], [self.l6E]])
    p20 = matrix([[0.0], [0.0], [self.l02]])

    p6E0 = RE0 * pE6
    p60 = pE0 - p6E0
    p260 = p60 - p20

    s = norm(p260)

    if s > self.l24 + self.l46:
      #print('invalid pose. L26 distance:{}'.format(s))
      return False

    q[3] = rs4 * (np.pi - np.arccos((self.l24**2 + self.l46**2 - s**2)/(2*self.l24 * self.l46)))
    q[2] = self.tr

    x = -cos(q[2])*sin(q[3]) * self.l46
    y = -sin(q[2])*sin(q[3]) * self.l46
    z = cos(q[3]) * self.l46 + self.l24
    xz = (x**2 + z **2) ** 0.5

    z_des = p260[2].item()

    q[1] = np.arccos(z_des / xz) - np.arctan2(x, z)
    if np.sign(q[1]) != rs2:
      q[1] = - np.arccos(z_des / xz) - np.arctan2(x, z)
    if np.sign(q[1]) != rs2:
      #print('Joint 2 has no solution for required {} sign.'.format('negative' if rs2 == -1 else 'positive'))
      #return False
      pass

    x = self.l24*sin(q[1]) + (cos(q[3])*sin(q[1]) - (cos(q[1])*cos(q[2]))*sin(q[3]))*self.l46

    x_des = p260[0].item()
    y_des = p260[1].item()

    q[0] = normalize_angle(np.arctan2(y_des, x_des) - np.arctan2(y,x))

    R20 = Ryz(q[1], q[0])
    R42 = Ryz(-q[3], q[2])
    R40 = R20 * R42
    p6E4 = R40.T * p6E0
    (q[5], q[4]) = rr(p6E4)
    if np.sign(q[5]) != rs6:
      q[4] = normalize_angle(q[4] + pi)
      q[5] = -q[5]
    
    R64 = Ryz(q[5], q[4])
    R60 = R40 * R64
    RE6 = R60.T * RE0
    q[6] = arctan2(RE6[1,0], RE6[0,0])

    if not self.checkJointLimits(q):
      return False
    return q

  def checkJointLimits(self, q):
        limit_exceeded = False
        for i, ti in enumerate(q):
          limit = self.joint_limits[self.joint_names[i]]
          if not (limit['lower'] <= ti <= limit['upper']):
            #print("{} with value {} is not within the joint limit range: {},{}".format(
                #self.joint_names[i], ti, limit['lower'], limit['upper']))
            limit_exceeded = True
        return not limit_exceeded

def normalize_angle(angle):
  while angle > pi:
    angle -= 2*pi
  while angle < -pi:
    angle += 2*pi
  return angle

def rotation_matrix(x,y,z):
  return rotate_x(x) @ rotate_y(y) @ rotate_z(z)

def rotate_x(q):
  (s, c) = (sin(q), cos(q))
  return matrix([[ 1.0, 0.0, 0.0],
                 [ 0.0,   c,  -s],
                 [ 0.0,   s,   c]])

def rotate_y(q):
  (s, c) = (sin(q), cos(q))  
  return matrix([[   c, 0.0,  -s],
                 [ 0.0, 1.0, 0.0],
                 [   s, 0.0,   c]])

def rotate_z(q):
  (s, c) = (sin(q), cos(q))
  return matrix([[   c,  -s, 0.0],
                 [   s,   c, 0.0],
                 [ 0.0, 0.0, 1.0]])

def Ryz(ty, tz):
  (cy, sy) = (cos(ty), sin(ty))
  (cz, sz) = (cos(tz), sin(tz))
  return matrix([[cy * cz, -sz, sy * cz],
                 [cy * sz, cz, sy * sz],
                 [-sy, 0.0, cy]])

def rr(p):
  ty = arctan2(sqrt(p[0,0]**2 + p[1,0]**2), p[2,0])
  tz = arctan2(p[1,0], p[0,0])

  if tz < -pi/2.0:
    ty = -ty
    tz += pi
  elif tz > pi/2.0:
    ty = -ty
    tz -= pi

  return (ty, tz)