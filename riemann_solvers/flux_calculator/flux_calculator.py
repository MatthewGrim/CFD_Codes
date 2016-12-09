"""
Author: Rohan
Date: 04/10/16

This file contains a a class providing methods to calculate the fluxes using Riemann solver based methods 
"""

from CFD_Projects.riemann_solvers.eos.thermodynamic_state import ThermodynamicState1D
from CFD_Projects.riemann_solvers.flux_calculator.riemann_solver import RiemannSolver
from CFD_Projects.riemann_solvers.flux_calculator.van_der_corput import VanDerCorput

import numpy as np


class FluxCalculatorND(object):
    GODUNOV = 1
    RANDOM_CHOICE = 2


class FluxCalculator1D(FluxCalculatorND):
    def __init__(self):
        pass
    
    @staticmethod
    def calculate_godunov_fluxes(densities, pressures, velocities, gamma):
        """
        Function used to calculate fluxes for a 1D simulation using Godunov's scheme as in Toro Chapter 6
        """
        density_fluxes = np.zeros(len(densities) - 1)
        momentum_fluxes = np.zeros(len(densities) - 1)
        total_energy_fluxes = np.zeros(len(densities) - 1)

        solver = RiemannSolver(gamma)

        for i, dens_flux in enumerate(density_fluxes):
            # Generate left and right states from cell averaged values
            left_state = ThermodynamicState1D(pressures[i], densities[i], velocities[i], gamma)
            right_state = ThermodynamicState1D(pressures[i + 1], densities[i + 1], velocities[i + 1], gamma)

            # Solve Riemann problem for star states
            p_star, u_star = solver.get_star_states(left_state, right_state)

            # Calculate fluxes using solver sample function
            p_flux, u_flux, rho_flux, _ = solver.sample(0.0, left_state, right_state, p_star, u_star)

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
        density_fluxes = np.zeros(len(densities) - 1)
        momentum_fluxes = np.zeros(len(densities) - 1)
        total_energy_fluxes = np.zeros(len(densities) - 1)

        solver = RiemannSolver(gamma)
        theta = VanDerCorput.calculate_theta(ts, 2, 1)
        for i in range(len(densities) - 2):
            # Generate left and right states from cell averaged values
            left_state = ThermodynamicState1D(pressures[i], densities[i], velocities[i], gamma)
            mid_state = ThermodynamicState1D(pressures[i + 1], densities[i + 1], velocities[i + 1], gamma)
            right_state = ThermodynamicState1D(pressures[i + 2], densities[i + 2], velocities[i + 2], gamma)

            # Solve Riemann problem for star states on either side of the cell
            p_star_left, u_star_left = solver.get_star_states(left_state, mid_state)
            p_star_right, u_star_right = solver.get_star_states(mid_state, right_state)

            # Calculate fluxes using solver sample function
            if theta <= 0.5:
                p_flux, u_flux, rho_flux, _ = solver.sample(theta * dx_over_dt, left_state, mid_state,
                                                         p_star_left, u_star_left)
            else:
                p_flux, u_flux, rho_flux, _ = solver.sample((theta - 1) * dx_over_dt, mid_state, right_state,
                                                         p_star_right, u_star_right)

            # Store fluxes in array
            density_fluxes[i] = rho_flux
            momentum_fluxes[i] = rho_flux * u_flux
            total_energy_fluxes[i] = p_flux / (left_state.gamma - 1) + 0.5 * rho_flux * u_flux * u_flux

        return density_fluxes, momentum_fluxes, total_energy_fluxes


class FluxCalculator2D(FluxCalculatorND):
    def __init__(self):
        pass

    @staticmethod
    def calculate_godunov_fluxes(densities, pressures, vel_x, vel_y, gamma):
        """
        Function used to calculate fluxes for a 2D simulation using Godunov's scheme
        """
        density_fluxes = np.zeros((densities.shape[0] - 1, densities.shape[1] - 1, 2))
        momentum_flux_x = np.zeros(density_fluxes.shape)
        momentum_flux_y = np.zeros(density_fluxes.shape)
        total_energy_fluxes = np.zeros(density_fluxes.shape)

        solver = RiemannSolver(gamma)

        i_length, j_length = np.shape(densities)
        for i in range(i_length - 1):
            for j in range(j_length - 1):
                # Generate left and right states from cell averaged values
                left_state = ThermodynamicState1D(pressures[i, j], densities[i, j], vel_x[i, j], gamma)
                right_state = ThermodynamicState1D(pressures[i + 1, j], densities[i + 1, j], vel_x[i + 1, j], gamma)

                # Solve Riemann problem for star states
                p_star, u_star = solver.get_star_states(left_state, right_state)

                # Calculate fluxes using solver sample function
                p_flux, u_flux, rho_flux, is_left = solver.sample(0.0, left_state, right_state, p_star, u_star)

                # Store fluxes in array
                v_y = vel_y[i, j] if is_left else vel_y[i + 1, j]
                density_fluxes[i, j - 1, 0] = rho_flux * u_flux
                momentum_flux_x[i, j - 1, 0] = rho_flux * u_flux * u_flux + p_flux
                momentum_flux_y[i, j - 1, 0] = rho_flux * u_flux * v_y
                e_tot = p_flux / (left_state.gamma - 1) + 0.5 * rho_flux * u_flux * u_flux + 0.5 * rho_flux * v_y ** 2
                total_energy_fluxes[i, j - 1, 0] = (p_flux + e_tot) * u_flux

                # Generate left and right states from cell averaged values
                left_state = ThermodynamicState1D(pressures[i, j], densities[i, j], vel_y[i, j], gamma)
                right_state = ThermodynamicState1D(pressures[i, j + 1], densities[i, j + 1], vel_y[i, j + 1], gamma)

                # Solve Riemann problem for star states
                p_star, v_star = solver.get_star_states(left_state, right_state)

                # Calculate fluxes using solver sample function
                p_flux, v_flux, rho_flux, is_left = solver.sample(0.0, left_state, right_state, p_star, v_star)

                # Store fluxes in array
                v_x = vel_x[i, j] if is_left else vel_x[i, j + 1]
                density_fluxes[i - 1, j, 1] = rho_flux * v_flux
                momentum_flux_x[i - 1, j, 1] = rho_flux * v_x * v_flux
                momentum_flux_y[i - 1, j, 1] = rho_flux * v_flux * v_flux + p_flux
                e_tot = p_flux / (left_state.gamma - 1) + 0.5 * rho_flux * v_flux * v_flux + 0.5 * rho_flux * v_x ** 2
                total_energy_fluxes[i - 1, j, 1] = (p_flux + e_tot) * v_flux

        return density_fluxes, momentum_flux_x, momentum_flux_y, total_energy_fluxes
