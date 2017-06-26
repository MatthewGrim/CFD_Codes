"""
Author: Rohan
Date: 24/06/17

This file contains a linear advection flux calculator that contains implementations of simple finite volume flux
calculator schemes
"""

import numpy as np


class AdvectionFluxCalculator(object):
    def __init__(self):
        pass

    @staticmethod
    def evaluate_fluxes(grid, dx, dt, a=None):
        """
        Function to evaluate fluxes on a given grid
        :param grid: The state grid
        :param a: Characteristic propagation speed
        :param dx: Grid spacing
        :param dt: Time spacing
        """
        raise NotImplementedError()


class LaxFriedrichsFluxCalculator(AdvectionFluxCalculator):
    """
    This flux calculator uses the integral form of the Lax Friedrichs finite difference advection scheme
    """
    def __init__(self):
        super(LaxFriedrichsFluxCalculator, self).__init__()

    @staticmethod
    def evaluate_fluxes(grid, dx, dt, a=None):
        # Loop through grid points generating fluxes
        fluxes = np.zeros(grid.shape[0] + 1)
        for i, flux in enumerate(fluxes):
            if i == 0:
                u_minus_1 = grid[i]
                u_0 = u_minus_1
            elif i == grid.shape[0]:
                u_minus_1 = grid[i - 1]
                u_0 = u_minus_1
            else:
                u_minus_1 = grid[i - 1]
                u_0 = grid[i]

            if a is not None:
                fluxes[i] = 0.5 * a * (u_0 + u_minus_1)
                fluxes[i] += 0.5 * dx / dt * (u_minus_1 - u_0)
            else:
                fluxes[i] = 0.25 * (u_0 ** 2 + u_minus_1 ** 2)
                fluxes[i] += 0.5 * dx / dt * (u_minus_1 - u_0)

        return fluxes


class LaxWendroffFluxCalculator(AdvectionFluxCalculator):
    """
    This calculator uses a two-step flux calculation, using half step states generated at the face centres
    to generate the fluxes for the time step. This is a leap frog scheme.
    """
    def __init__(self):
        super(LaxWendroffFluxCalculator, self).__init__()

    @staticmethod
    def evaluate_fluxes(grid, dx, dt, a=None):
        # Loop through grid points generating half time step states on staggered grid
        staggered_grid = np.zeros(grid.shape[0] + 1)
        for i, u_stag in enumerate(staggered_grid):
            if i == 0:
                u_minus_1 = grid[i]
                u_0 = u_minus_1
            elif i == grid.shape[0]:
                u_minus_1 = grid[i - 1]
                u_0 = u_minus_1
            else:
                u_minus_1 = grid[i - 1]
                u_0 = grid[i]

            if a is not None:
                staggered_grid[i] = 0.5 * a * dt / dx * (u_minus_1 - u_0)
                staggered_grid[i] += 0.5 * (u_minus_1 + u_0)
            else:
                staggered_grid[i] = 0.25 * dt / dx * (u_minus_1 ** 2 - u_0 ** 2)
                staggered_grid[i] += 0.5 * (u_minus_1 + u_0)

        # Calculate fluxes from staggered grid states
        fluxes = np.zeros(grid.shape[0] + 1)
        for i, flux in enumerate(fluxes):
            if a is not None:
                fluxes[i] = a * staggered_grid[i]
            else:
                fluxes[i] = 0.5 * staggered_grid[i] ** 2

        return fluxes
