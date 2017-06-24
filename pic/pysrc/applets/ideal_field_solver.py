"""
Author: Rohan
Date: 27/03/17

This file contains a solver to obtain the analytic solution to a 3D magnetic field on a single charged particle,
with a given velocity and charge.
"""

from CFD_Projects.pic.pysrc.applets.vector_ops import *

import numpy as np


class ChargedParticle(object):
    """
    A charge particle that acts as a struct to store the properties of the charged particle
    """
    def __init__(self, mass, charge, position, velocity):
        assert isinstance(mass, float), "mass must be a float"
        assert isinstance(charge, float), "charge must be a float"
        assert isinstance(velocity, np.ndarray) and velocity.shape[0] == 3 and len(velocity.shape) == 1, \
            "velocity must be a 3D vector"
        assert isinstance(position, np.ndarray) and position.shape[0] == 3 and len(position.shape) == 1, \
            "position must be a 3D vector"

        self._mass = mass
        self._charge = charge
        self.position = position
        self.velocity = velocity

    @property
    def mass(self):
        return self._mass

    @property
    def charge(self):
        return self._charge


def solve_field(particle, B):
    """
    Solve for the velocity function with respect to time
    """
    assert isinstance(B, np.ndarray) and B.shape[0] == 3 and len(B.shape) == 1, "B must be a 3D vector"
    v = particle.velocity
    omega = particle.charge * magnitude(B) / particle.mass
    v_parallel = vector_projection(v, B)
    v_perpendicular = v - v_parallel

    F = cross(v_perpendicular, B) * particle.charge
    radius = magnitude(F) / (particle.mass * dot(v_perpendicular, v_perpendicular))
    centre_of_rotation = F * radius / magnitude(F) + particle.position

    def parallel_motion(t):
        return v_parallel * t

    def perpendicular_motion(t):
        angle = omega * t
        return rotate_3d()


if __name__ == '__main__':
    particle = ChargedParticle(1.0, 2.0, np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 1.0, 1.0]))

    print(particle.mass)
    print(particle.charge)
    print(particle.position)
    print(particle.velocity)

    print(magnitude(particle.velocity))
    print(dot(particle.velocity, particle.position))