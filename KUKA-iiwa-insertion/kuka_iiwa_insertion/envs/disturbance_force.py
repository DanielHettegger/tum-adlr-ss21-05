import numpy as np
from numpy.linalg import norm

class DisturbanceForce():
    def __init__(self):
        pass
    
    def generate(self):
        raise NotImplementedError

    def get_force(self, position):
        raise NotImplementedError

class StaticForce(DisturbanceForce):
    def __init__(self, max_force=0.1, direction=None, magnitude=None):
        self.magnitude = magnitude
        self.direction = direction
        self.max_force = max_force
        self.generate()

    def generate(self):
        if self.direction is not None:
            self.force = self.direction
        else:
            self.force = get_random_vector()

        if self.magnitude is not None:
            self.force *= self.magnitude / norm(self.force)
        else:
            self.force *= self.max_force

    def get_force(self, position):
        return self.force
            

class SpringForceXY(DisturbanceForce):
    def __init__(self, max_magnitude=0.1, distance=None, direction=None, spring_konstant=None, position=None, center_position=[0.6,0.0]): # center position
        self.max_magnitude = max_magnitude
        self.distance = distance
        self.direction = direction
        self.spring_konstant = spring_konstant
        self.sample_spring_konstant = spring_konstant is None
        self.position = position
        self.center_position = np.array(center_position)
        self.generate()

    def generate(self):
        if self.direction is not None:
            position = direction * np.random.rand(1)
        else:
            position = get_random_vector(2)

        if self.distance is not None:
            position *= self.distance / norm(position)
        
        else:
            position *= self.max_magnitude

        if self.position is not None:
            self.resting_position = self.position
        else:
            self.resting_position = position

        if self.sample_spring_konstant:
            self.spring_konstant = np.random.rand(1) * 0.2 + 0.1
    
    def get_force(self, position):
        position = np.array(position[:2])
        force =  (self.resting_position - (position - self.center_position)) * self.spring_konstant
        return np.append(force, [0.0])        
                


def get_random_vector(dim=3):
    candidate = np.inf
    while not norm(candidate) <= 1:
        candidate = np.random.rand(dim) * 2 - 1
    return candidate