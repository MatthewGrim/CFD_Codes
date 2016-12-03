"""
Author: Rohan
Date: 04/10/16

This file contains a a class providing methods to calculate the fluxes using Riemann solver based methods 
"""

from CFD_Projects.riemann_solvers.eos.thermodynamic_state import ThermodynamicState
from CFD_Projects.riemann_solvers.flux_calculator.riemann_solver import RiemannSolver
from CFD_Projects.riemann_solvers.flux_calculator.van_der_corput import VanDerCorput

import numpy as np


class FluxCalculator(object):
    GODUNOV = 1
    RANDOM_CHOICE = 2

    def __init__(self):
        pass
    
    @staticmethod
    def calculate_godunov_fluxes(densities, pressures, velocities, gamma):
        """
        Function used to calculate fluxes for a 1D simulation using Godunov's scheme as in Toro Chapter 6
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

    @staticmethod
    def calculate_random_choice_fluxes(densities, pressures, velocities, gamma, ts, dx_over_dt):
        """
        Function used to calculate states for a 1D simulation using Glimm's random choice scheme as in Toro Chapter 7
        """
        density_fluxes = np.zeros(len(densities) + 1)
        momentum_fluxes = np.zeros(len(densities) + 1)
        total_energy_fluxes = np.zeros(len(densities) + 1)

        solver = RiemannSolver(gamma)
        theta = VanDerCorput.calculate_theta(ts, 2, 1)
        for i, dens in enumerate(densities):
            # Generate left and right states from cell averaged values
            if i == 0:
                left_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
                mid_state = left_state
                right_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
            elif i == len(densities) - 1:
                left_state = ThermodynamicState(pressures[i - 2], densities[i - 2], velocities[i - 2], gamma)
                mid_state = ThermodynamicState(pressures[i - 1], densities[i - 1], velocities[i - 1], gamma)
                right_state = mid_state
            else:
                left_state = ThermodynamicState(pressures[i - 1], densities[i - 1], velocities[i - 1], gamma)
                mid_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
                right_state = ThermodynamicState(pressures[i + 1], densities[i + 1], velocities[i + 1], gamma)

            # Solve Riemann problem for star states on either side of the cell
            p_star_left, u_star_left = solver.get_star_states(left_state, mid_state)
            p_star_right, u_star_right = solver.get_star_states(mid_state, right_state)

            # Calculate fluxes using solver sample function
            if theta <= 0.5:
                p_flux, u_flux, rho_flux = solver.sample(theta * dx_over_dt, left_state, mid_state,
                                                         p_star_left, u_star_left)
            else:
                p_flux, u_flux, rho_flux = solver.sample((theta - 1) * dx_over_dt, mid_state, right_state,
                                                         p_star_right, u_star_right)

            # Store fluxes in array
            density_fluxes[i] = rho_flux
            momentum_fluxes[i] = rho_flux * u_flux
            total_energy_fluxes[i] = p_flux / (left_state.gamma - 1) + 0.5 * rho_flux * u_flux * u_flux

        return density_fluxes, momentum_fluxes, total_energy_fluxes

    @staticmethod
    def calculate_muscl_hancock_fluxes(densities, pressures, velocities, gamma, dx):
        """
        Function to calculate the fluxes in a 1D cell using the MUSCL Hancock scheme in Toro Chapter 14
        """
        density_fluxes = np.zeros(len(densities) + 1)
        momentum_fluxes = np.zeros(len(densities) + 1)
        total_energy_fluxes = np.zeros(len(densities) + 1)

        solver = RiemannSolver(gamma)

        for i, dens_flux in enumerate(density_fluxes):
            # Generate left and right states from cell averaged values
            if i == 0:
                second_left_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
                left_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
                mid_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
                right_state = ThermodynamicState(pressures[i + 1], densities[i + 1], velocities[i + 1], gamma)
                second_right_state = ThermodynamicState(pressures[i + 2], densities[i + 2], velocities[i + 2], gamma)
            elif i == 1:
                second_left_state = ThermodynamicState(pressures[i - 1], densities[i - 1], velocities[i - 1], gamma)
                left_state = ThermodynamicState(pressures[i - 1], densities[i - 1], velocities[i - 1], gamma)
                mid_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
                right_state = ThermodynamicState(pressures[i + 1], densities[i + 1], velocities[i + 1], gamma)
                second_right_state = ThermodynamicState(pressures[i + 2], densities[i + 2], velocities[i + 2], gamma)

            elif i == len(density_fluxes) - 2:
                second_left_state = ThermodynamicState(pressures[i - 2], densities[i - 2], velocities[i - 2], gamma)
                left_state = ThermodynamicState(pressures[i - 1], densities[i - 1], velocities[i - 1], gamma)
                mid_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
                right_state = ThermodynamicState(pressures[i + 1], densities[i + 1], velocities[i + 1], gamma)
                second_right_state = ThermodynamicState(pressures[i + 1], densities[i + 1], velocities[i + 1], gamma)

            elif i == len(density_fluxes) - 1:
                second_left_state = ThermodynamicState(pressures[i - 2], densities[i - 2], velocities[i - 2], gamma)
                left_state = ThermodynamicState(pressures[i - 1], densities[i - 1], velocities[i - 1], gamma)
                mid_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
                right_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
                second_right_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
            else:
                second_left_state = ThermodynamicState(pressures[i - 2], densities[i - 2], velocities[i - 2], gamma)
                left_state = ThermodynamicState(pressures[i - 1], densities[i - 1], velocities[i - 1], gamma)
                mid_state = ThermodynamicState(pressures[i], densities[i], velocities[i], gamma)
                right_state = ThermodynamicState(pressures[i + 1], densities[i + 1], velocities[i + 1], gamma)
                second_right_state = ThermodynamicState(pressures[i + 2], densities[i + 2], velocities[i + 2], gamma)

            # Calculate density, momentum and total energy gradients

            # Find UL and UR for left, mid and right states by applying gradients

            # Apply half time step update to get ULbar and URbar

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