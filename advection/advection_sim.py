"""
Author: Rohan
Date: 24/06/17

This file contains a simulation class for running 1D linear advection equation simulations
"""

import numpy as np
from CFD_Projects.advection.flux_calculator import AdvectionFluxCalculator


class LinearAdvectionSim(object):
    """
    This class is used to run simple linear advection simulations on a uniform one dimensional.
    """
    def __init__(self, x, initial_grid, flux_calculator_model, final_time, num_pts=1000, a=None):
        """
        Constuctor for linear advection simulation solving:

        Ut + aUx = 0

        :param x: Grid points - assumed to be uniform
        :param initial_grid: Values of state u on grid
        :param flux_calculator_model: Flux calculator method used in simulation
        :param final_time: Final time of the simulation
        :param num_pts: Number of points in time
        """
        assert isinstance(final_time, float) and final_time > 0.0
        assert isinstance(flux_calculator_model, AdvectionFluxCalculator)
        assert isinstance(x, np.ndarray)
        assert isinstance(initial_grid, np.ndarray)
        assert x.shape == initial_grid.shape

        self.flux_calculator = flux_calculator_model
        self.x = x
        self.u = initial_grid
        self.final_time = final_time
        self.times = np.linspace(0.0, final_time, num_pts)
        self.dt = self.times[1] - self.times[0]
        self.dx = self.x[1] - self.x[0]
        self.a = a

    def update_states(self, fluxes):
        """
        Function to update the states on the grid based on fluxes between cells
        """
        for i, state in enumerate(self.u):
            self.u[i] += fluxes[i] - fluxes[i + 1]

    def run_simulation(self):
        """
        Main function to run simulation and return final grid state
        """
        for i, t in enumerate(self.times):
            fluxes = self.flux_calculator.evaluate_fluxes(self.u, self.dx. self.dt, a=self.a)
            self.update_states(fluxes)

        return self.x, self.u