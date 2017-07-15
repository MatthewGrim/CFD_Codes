"""
Author: Rohan Ramasamy
Date: 15/07/17

This file contains the implementation of the boris method for updating the position and velocities of charged particles
in an electromagnetic field.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pic.pysrc.geometry.vector_ops import *
from pic.pysrc.simulations.analytic_single_particle_motion import solve_B_field, ChargedParticle


def boris_solver(E_field, B_field, X, V, Q, M, dt):
    """
    Function to update the positon of a set of particles in an electromagnetic field over the time dt

    :param E_field: function to evaluate the 3D E field at time t
    :param B_field: function to evaluate the 3D B field at time t
    :param X: position of the particles in the simulation domain
    :param V: velocities of the particles in the simulation domain
    :param Q: charges of the particles in the simulation domain
    ;param M: masses of the particles in the simulation domain
    :return:
    """
    assert isinstance(X, np.ndarray) and X.shape[1] == 3
    assert isinstance(V, np.ndarray) and V.shape[1] == 3
    assert X.shape[0] == V.shape[0] == Q.shape[0] == M.shape[0]
    assert isinstance(dt, float)

    # Calculate v minus
    E_field_offset = Q * E_field(X) / M
    v_minus = V + E_field_offset

    # Calculate v prime
    t = Q * B_field(X) / M * 0.5 * dt
    v_prime = np.zeros(v_minus.shape)
    for i, v in enumerate(v_prime[:, 0]):
        v_prime[i, :] = v_minus[i, :] + cross(v_minus[i, :], t[i, :])

    # Calculate v plus
    s = np.zeros(t.shape)
    for i, _ in enumerate(t[:, 0]):
        s[i, :] = 2 * t[i, :]
        s[i, :] /= 1 + magnitude(t[i, :]) ** 2

    v_plus = np.zeros(v_minus.shape)
    for i, v in enumerate(v_prime[:, 0]):
        v_plus[i, :] = v_minus[i, :] + cross(v_prime[i, :], s[i, :])

    # Calculate new velocity
    V_plus = v_plus + E_field_offset

    # Integrate to get new positions
    X_plus = X + V_plus * dt

    return X_plus, V_plus


def single_particle_example():
    """
    Example solution to simple magnetic field case
    :return:
    """
    def B_field(x):
        B = np.zeros(x.shape)
        for i, b in enumerate(B):
            B[i, :] = np.asarray([0.0, 0.0, 1.0])
        return B

    def E_field(x):
        E = np.zeros(x.shape)
        return E

    X = np.asarray([[0.0, 1.0, 0.0]])
    V = np.asarray([[1.0, 0.0, 0.0]])
    Q = np.asarray([1.0])
    M = np.asarray([1.0])

    times = np.linspace(0.0, 4.0, 1000)
    positions = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    velocities = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    for i, t in enumerate(times):
        if i == 0:
            positions[i, :, :] = X
            velocities[i, :, :] = V
            continue

        dt = times[i] - times[i - 1]

        x, v = boris_solver(E_field, B_field, X, V, Q, M, dt)

        positions[i, :, :] = x
        velocities[i, :, :] = v
        X = x
        V = v

    particle = ChargedParticle(1.0, 1.0, np.asarray([0.0, 1.0, 0.0]), np.asarray([1.0, 0.0, 0.0]))
    B = np.asarray([0.0, 0.0, 1.0])
    analytic_times, analytic_positions = solve_B_field(particle, B, 4.0)

    x = positions[:, :, 0].flatten()
    y = positions[:, :, 1].flatten()
    z = positions[:, :, 2].flatten()
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot('111', projection='3d')
    ax.plot(x, y, z, label='numerical')
    ax.plot(analytic_positions[:, 0], analytic_positions[:, 1], analytic_positions[:, 2], label='analytic')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='best')
    ax.set_title("Analytic and Numerical Particle Motion")
    plt.show()

    fig, axes = plt.subplots(3, figsize=(20, 10))
    axes[0].plot(x - analytic_positions[:, 0])
    axes[1].plot(y - analytic_positions[:, 1])
    axes[2].plot(z - analytic_positions[:, 2])
    fig.suptitle("Deviation of Numerical Solution from the Analytic")
    plt.show()

if __name__ == '__main__':
    single_particle_example()

