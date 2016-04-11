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
        :return: a 3 dimensional array containing the spatial solution for each
                time step
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

    @staticmethod
    def laplace_2d(x_loc, y_loc, u):
        """
        A function to solve the two dimensional Laplace equation

                        d2u/dx2 + d2u/dy2 = 0
        :param x_loc: A 1D array of the x coordinates in the system
        :param y_loc: A 1D array of the y coordinates in the system
        :param u: A 2D array of variable states in the system
        :return: A two dimensional array containing the iterative solution to the system given
                the boundary conditions specified in the function.
        """

        assert u.shape == (x_loc.shape[0], y_loc.shape[0])

        dx = x_loc[1:] - x_loc[0:-1]
        dy = y_loc[1:] - y_loc[0:-1]

        u_sol = u.copy()
        error = 1
        while error > 1e-3:
            u = u_sol.copy()
            u_sol[1:-1, 1:-1] = (dy[0:-1] * dy[1:] * (u_sol[2:, 1:-1] + u_sol[0:-2, 1:-1]) \
                                + dx[0:-1] * dx[1:] * (u_sol[1:-1, 2:] + u_sol[1:-1, 0:-2])) \
                                / (2 * (dy[0:-1] * dy[1:] + dx[0:-1] * dx[1:]))

            u_sol[0, :] = 0
            u_sol[-1, :] = y_loc
            u_sol[:, 0] = u_sol[:, 1]
            u_sol[:, -1] = u_sol[:, -2]

            error = np.abs(np.sum(np.abs(u_sol[:]) - np.abs([u])) / np.sum(np.abs(u[:])))
        return u_sol

    @staticmethod
    def poisson_2d(x_loc, y_loc, u, b, num_iter):
        """
        Function to iteratively solve the Poisson equation:
                    d2u/dx2 + d2u/dy2 = b

        :param x_loc: location of x positions
        :param y_loc: location of y positions
        :param u: state variable being iteratively solved
        :param b: source term
        :param num_iter: number of iterations made before returning solution
        :return: a 2D array giving the solution for the boundary conditions specified in this
                function
        """
        assert u.shape == (x_loc.shape[0], y_loc.shape[0])
        assert b.shape == (x_loc.shape[0], y_loc.shape[0])

        dx = x_loc[1:] - x_loc[0:-1]
        dy = y_loc[1:] - y_loc[0:-1]

        u_sol = u.copy()
        for i in range(num_iter):
            u_sol[1:-1, 1:-1] = (dy[0:-1] * dy[1:] * (u_sol[2:, 1:-1] + u_sol[0:-2, 1:-1]) \
                                + dx[0:-1] * dx[1:] * (u_sol[1:-1, 2:] + u_sol[1:-1, 0:-2]) \
                                 - (dy[0:-1] * dy[1:] * dx[0:-1] * dx[1:] * b[1:-1, 1:-1])) \
                                / (2 * (dy[0:-1] * dy[1:] + dx[0:-1] * dx[1:]))

            u_sol[0, :] = 0
            u_sol[-1, :] = 0
            u_sol[:, 0] = 0
            u_sol[:, -1] = 0

        return u_sol
