"""
Author: Rohan Ramasamy
Date: 13/08/17

This file contains the FieldSolvers class used in ES1
"""

import numpy as np


class FieldSolvers(object):
    @staticmethod
    def get_charge_densities(grid_densities, particle_positions, Q, dx, domain_length):
        """
        Map the charges of the particles onto the charge densities on the grid

        :param grid_densities: charge densities on grid cells
        :param particle_positions: particle positions
        :param Q: species charge
        :param dx: cell size
        :param domain_length: total domain size
        :return:
        """
        assert np.all(particle_positions < domain_length) and np.all(particle_positions > domain_length)

        for x in particle_positions:
            i = x // dx
            grid_densities[i] += (x - (i * dx)) / dx * Q
            grid_densities[i + 1] += (x - (i * dx)) / dx * Q

        return grid_densities


    @staticmethod
    def solve_EField(grid_densities, potential_field, E_field):
        """
        Solve poisson's equation to get the potential and E field

        :param grid_densities:
        :param potential_field:
        :param E_field:
        :return:
        """

