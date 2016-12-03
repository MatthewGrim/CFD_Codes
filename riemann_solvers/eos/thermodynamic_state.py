"""
Author: Rohan
Date: 29/06/16

This file contains a class used to model a numerical solution to Riemann problems. The implementation of the Riemann
follows that in Toro - Chapter 4.
"""

import numpy as np


class ThermodynamicState(object):
    def __init__(self, pressure, density, velocity, gamma):
        assert isinstance(pressure, float)
        assert isinstance(density, float)
        assert isinstance(velocity, float)
        assert isinstance(gamma, float)

        self.gamma = gamma
        self.p = pressure
        self.rho = density
        self.u = velocity

        self.mom = self.u * self.rho
        self.e_kin = 0.5 * self.rho * self.u * self.u
        self.e_int = pressure / (self.rho * (gamma - 1))

    def sound_speed(self):
        return np.sqrt(self.gamma * self.p / self.rho)

    def update_states(self, density_flux, momentum_flux, e_flux, i):
        """
        Updates the thermodynamic state of the cell based on conservative fluxes into volume

        :param density_flux
        :param momentum_flux
        :param e_flux
        """
        assert(isinstance(density_flux, float))
        assert(isinstance(momentum_flux, float))
        assert(isinstance(e_flux, float))

        e_tot_initial = self.rho * self.e_int + self.e_kin + e_flux

        self.rho += density_flux
        self.mom += momentum_flux

        self.u = self.mom / self.rho

        self.e_kin = 0.5 * self.rho * self.u ** 2
        self.e_int = (e_tot_initial - self.e_kin) / self.rho

        self.p = self.rho * self.e_int * (self.gamma - 1)

