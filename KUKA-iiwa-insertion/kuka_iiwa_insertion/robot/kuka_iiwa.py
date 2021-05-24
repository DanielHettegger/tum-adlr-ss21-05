#taken and modified from:
#https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kuka.py
import os, inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(os.path.dirname(currentdir))
#os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
from numpy import pi, sqrt, cos, sin, arctan2, array, matrix
from numpy.linalg import norm
#import copy
import math
import pybullet_data
from ..files import get_resource_path

from scipy.spatial.transform import Rotation as R


class KukaIIWA:
  def __init__(self,client):
    self.client = client

    self.ee_index = 8

    self.l02 = 0.36
    self.l24 = 0.42
    self.l46 = 0.4
    self.l6E = 0.126 + 0.026

    self.robot_name = 'iiwa'

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

    self.kuka_uid = p.loadURDF(get_resource_path('kuka_iiwa_insertion','robot', 'urdf', 'iiwa14.urdf'), physicsClientId=self.client)
    self.reset()

  def reset(self):
    self.rs = 2
    self.tr = 0.0

    self.target_position =    [0.7, 0, 0.4]
    self.target_orientation = [pi, 0, pi]

    
    q = self.inverse_kinematics(self.target_position, self.target_orientation)
    if q:
      self.force_joint_targets(q)
  
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

  def apply_action(self, action):
    target_candidate = self.target_position        
    target_candidate[0] += action[0]
    target_candidate[1] += action[1]
    target_candidate[2] += action[2]
    #self.target_orientation[2] += action[3]

    
    q = self.inverse_kinematics(target_candidate, self.target_orientation) 
    if q:
      self.set_joint_targets(q)
      self.target_position = target_candidate

  def set_joint_targets(self, q):
    for i, qi in enumerate(q):
      p.setJointMotorControl2(bodyUniqueId= self.kuka_uid,
                              jointIndex= i+1,
                              controlMode= p.POSITION_CONTROL,
                              targetPosition= qi,
                              targetVelocity=0,
                              #force=self.maxForce,
                              #maxVelocity=self.maxVelocity,
                              positionGain=0.3,
                              velocityGain=1)
  
  def force_joint_targets(self, q):
    for i, qi in enumerate(q):
      p.resetJointState(bodyUniqueId= self.kuka_uid,
                        jointIndex= i+1,
                        targetValue= qi,
                        physicsClientId= self.client)


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