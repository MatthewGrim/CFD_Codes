"""
Author: Rohan
Date: 27/03/17

This file contains a solver to obtain the analytic solution to a 3D magnetic field on a single charged particle,
with a given velocity and charge.
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from CFD_Projects.pic.pysrc.geometry.vector_ops import *


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


def solve_B_field(particle, B, final_time, num_pts=1000):
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
    relative_position = particle.position - centre_of_rotation

    def parallel_motion(t):
        return v_parallel * t

    def perpendicular_motion(t):
        angle = -omega * t
        return arbitrary_axis_rotation_3d(relative_position, B, angle) - particle.position

    times = np.linspace(0.0, final_time, num_pts)
    positions = np.zeros((num_pts, 3))
    for i, t in enumerate(times):
        positions[i, :] = particle.position + parallel_motion(t) + perpendicular_motion(t)

    return times, positions


def solve_E_field(particle, E, final_time, num_pts=1000):
    """
    Solve for the velocity function with respect to time
    """
    assert isinstance(E, np.ndarray) and E.shape[0] == 3 and len(E.shape) == 1, "B must be a 3D vector"
    F = E * particle.charge

    def E_field_motion(t):
        return 0.5 * F / particle.mass * t ** 2

    times = np.linspace(0.0, final_time, num_pts)
    positions = np.zeros((num_pts, 3))
    for i, t in enumerate(times):
        positions[i, :] = particle.position + particle.velocity * t + E_field_motion(t)

    return times, positions


def B_field_example():
    """
    Example B field solution
    :return:
    """
    particle = ChargedParticle(1.0, 2.0, np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 1.0]))
    B = np.asarray([0.0, 0.0, 1.0])

    times, positions = solve_B_field(particle, B, 3.0)

    fig = plt.figure()
    ax = fig.add_subplot('111', projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    plt.show()


def E_field_example():
    """
    Example E field solution
    :return:
    """
    particle = ChargedParticle(1.0, 2.0, np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0]))
    E = np.asarray([0.0, 0.0, 3.0])

    times, positions = solve_E_field(particle, E, 3.0)

    fig = plt.figure()
    ax = fig.add_subplot('111', projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    plt.show()


if __name__ == '__main__':
    B_field_example()
    # E_field_example()