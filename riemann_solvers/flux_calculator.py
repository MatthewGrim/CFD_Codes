"""
Author: Rohan
Date: 04/10/16

This file contains a a class providing methods to calculate the fluxes using Riemann solver based methods 
"""

from CFD_Projects.riemann_solvers.thermodynamic_state import ThermodynamicState
from CFD_Projects.riemann_solvers.riemann_solver import RiemannSolver

import numpy as np


class FluxCalculator(object):
    def __init__(self):
        pass
    
    @staticmethod
    def calculate_godunov_fluxes(densities, pressures, velocities, gamma):
        """
        Function used to calculate first order godunov fluxes for a 1D simulation
        """
        density_fluxes = np.zeros(len(densities) + 1)
        momentum_fluxes = np.zeros(len(densities) + 1)
        total_energy_fluxes = np.zeros(len(densities) + 1)

        solver = RiemannSolver(gamma)

        for i, dens_flux in enumerate(density_fluxes):
            # Generate left and right states from cell averaged values
            if i == 0:
                left_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
                right_state = left_state
            elif i == len(density_fluxes) - 1:
                left_state = ThermodynamicState(pressures[i - 1], densities[i - 1], velocities[i - 1], gamma)
                right_state = left_state
            else:
                left_state = ThermodynamicState(pressures[i - 1], densities[i - 1], velocities[i - 1], gamma)
                right_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)

            # Solve Riemann problem for star states
            p_star, u_star = solver.get_star_states(left_state, right_state)

            # Calculate fluxes using solver sample function
            p_flux, u_flux, rho_flux = solver.sample(0.0, left_state, right_state, p_star, u_star)

            # Store fluxes in array
            density_fluxes[i] = rho_flux * u_flux
            momentum_fluxes[i] = rho_flux * u_flux * u_flux + p_flux
            e_tot = p_flux / (left_state.gamma - 1) + 0.5 * rho_flux * u_flux * u_flux
            total_energy_fluxes[i] = (p_flux + e_tot) * u_flux

        return density_fluxes, momentum_fluxes, total_energy_fluxes