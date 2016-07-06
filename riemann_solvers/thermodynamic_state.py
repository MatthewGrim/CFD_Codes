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

        self.a = np.sqrt(gamma * pressure / density)