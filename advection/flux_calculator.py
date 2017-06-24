"""
Author: Rohan
Date: 24/06/17

This file contains a linear advection flux calculator that contains implementations of simple finite volume flux
calculator schemes
"""


class AdvectionFluxCalculator(object):
    def __init__(self):
        raise NotImplementedError()

    def evaluate_fluxes(self, grid, dx, dt, a=None):
        """
        Function to evaluate fluxes on a given grid
        :param grid: The state grid
        :param a: Characteristic propagation speed
        :param dx: Grid spacing
        :param dt: Time spacing
        """
        raise NotImplementedError()

