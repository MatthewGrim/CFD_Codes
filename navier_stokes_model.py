"""
Author: Rohan
Date: 20/03/16
"""
# Standard imports
import numpy as np


class NavierStokes(object):
    def __init__(self):
        pass

    @staticmethod
    def convection_1d(x_loc, time_steps, initial_velocities, c=1, linear=True):
        """
        Function to solve the one dimensional convection equation:

        Linear:       du/dt + c * du/dx = 0
        Non-Linear:   du/dt + u * du/dx = 0

        :param x_loc: discrete cell locations being solved for
        :param time_steps: discrete time values being solved
        :param initial_velocities: initial velocity condition in domain
        :param c: wave speed
        :paramL linear: boolean to determine whether the solution is linear or not
        :return: a 2 dimensional array containing the spatial solution for each
                time step
        """

        assert initial_velocities.shape == x_loc.shape
        u_solution = np.zeros((x_loc.shape[0], time_steps.shape[0]))

        delta_t = time_steps[1:] - time_steps[0:-1]
        dx = x_loc[1:] - x_loc[0:-1]

        u_solution[:, 0] = initial_velocities
        i = 1
        while i < time_steps.shape[0]:
            dt = delta_t[i - 1]

            if linear is True:
                u_solution[1:, i] = u_solution[1:, i - 1] \
                                    - c * dt / dx * (u_solution[1:, i - 1] \
                                                        - u_solution[0:-1, i - 1])
            else:
                u_solution[1:, i] = u_solution[1:, i - 1] \
                                    - u_solution[1:, i - 1] * dt / dx * (u_solution[1:, i - 1] \
                                                                         - u_solution[0:-1, i - 1])

            # Apply boundary condition at first x_location
            u_solution[0, i] = u_solution[0, 0]
            i += 1

        return u_solution

    @staticmethod
    def convection_2d(x_loc, y_loc, time_steps, u_initial, v_initial, c=1, linear=True):
        """
        Function to solve the 2D convection equation:

                du/dt + c*du/dx + c*du/dy = 0

        :param x_loc: the location of x cell centres in the mesh
        :param y_loc: the location of y cell centres in the mesh
        :param time_steps: the time steps used in the simulation
        :param u_initial: the initial x velocity condition to solve for
        :param v_initial: the initial y velocity condition to solve for
        :param c: the wave speed
        :param linear: boolean to determin whether the equation is linear or not
        :return:
        """

        assert u_initial.shape == (x_loc.shape[0], y_loc.shape[0])
        assert v_initial.shape == (x_loc.shape[0], y_loc.shape[0])

        delta_t = time_steps[1:] - time_steps[0:-1]
        dx = x_loc[1:] - x_loc[0:-1]
        dy = y_loc[1:] - y_loc[0:-1]

        u_solution = np.zeros((time_steps.shape[0], x_loc.shape[0], y_loc.shape[0]))
        v_solution = np.zeros((time_steps.shape[0], x_loc.shape[0], y_loc.shape[0]))
        u_solution[0, :, :] = u_initial
        v_solution[0, :, :] = v_initial
        i=1
        while i < time_steps.shape[0]:
            dt = delta_t[i - 1]
            if linear:
                u_solution[i, 1:, 1:] = u_solution[i - 1, 1:, 1:] - \
                                        c*dt/dx*(u_solution[i - 1, 1:, 1:] -
                                                 u_solution[i - 1, :-1, 1:]) - \
                                        c*dt/dy*(u_solution[i - 1, 1:, 1:] -
                                                 u_solution[i - 1, 1:, :-1])
                v_solution[i, 1:, 1:] = v_solution[i - 1, 1:, 1:] - \
                                        c*dt/dx*(v_solution[i - 1, 1:, 1:] -
                                                 v_solution[i - 1, :-1, 1:]) - \
                                        c*dt/dy*(v_solution[i - 1, 1:, 1:] -
                                                 v_solution[i - 1, 1:, :-1])
            else:
                u_solution[i, 1:, 1:] = u_solution[i - 1, 1:, 1:] - \
                                        u_solution[i - 1, 1:, 1:] * dt/dx * \
                                        (u_solution[i - 1, 1:, 1:] -
                                         u_solution[i - 1, :-1, 1:]) - \
                                        v_solution[i - 1, 1:, 1:] * dt/dy * \
                                        (u_solution[i - 1, 1:, 1:] -
                                         u_solution[i - 1, 1:, :-1])
                v_solution[i, 1:, 1:] = v_solution[i - 1, 1:, 1:] - \
                                        u_solution[i - 1, 1:, 1:] * dt/dx * \
                                        (v_solution[i - 1, 1:, 1:] -
                                         v_solution[i - 1, :-1, 1:]) - \
                                        v_solution[i - 1, 1:, 1:] * dt/dy * \
                                        (v_solution[i - 1, 1:, 1:] -
                                         v_solution[i - 1, 1:, :-1])
            u_solution[i, 0, :] = 1
            u_solution[i, -1, :] = 1
            u_solution[i, :, 0] = 1
            u_solution[i, :, -1] = 1
            v_solution[i, 0, :] = 1
            v_solution[i, -1, :] = 1
            v_solution[i, :, 0] = 1
            v_solution[i, :, -1] = 1
            i += 1

        return u_solution, v_solution
