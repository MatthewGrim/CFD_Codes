"""
Author: Rohan
Date: 18/03/17

This file contains flux limiter classes for producing high order TVD methods
"""

import numpy as np


class BaseLimiter(object):
    @staticmethod
    def _calculate_slope_ratios(left_rho_slope, right_rho_slope,
                               left_mom_slope, right_mom_slope,
                               left_energy_slope, right_energy_slope):
        """
        Function used to calculate to ratio of the different conservative variables
        """
        TOL = 1e-14

        r_density = 0.0 if np.isclose(right_rho_slope, 0.0, atol=TOL) else left_rho_slope / right_rho_slope
        r_momentum = 0.0 if np.isclose(right_mom_slope, 0.0, atol=TOL) else left_mom_slope / right_mom_slope
        r_energy = 0.0 if np.isclose(right_energy_slope, 0.0, atol=TOL) else left_energy_slope / right_energy_slope

        return r_density, r_momentum, r_energy

    @staticmethod
    def _calculate_eta_values(c, r_density, r_momentum, r_energy):
        """
        Function to calculate max allowable slope values for the left and right states
        """
        beta_R = 1.0
        beta_L = 1.0
        eta_density_L = 2.0 * beta_L * r_density / (1 + r_density)
        eta_momentum_L = 2.0 * beta_L * r_momentum / (1 + r_momentum)
        eta_energy_L = 2.0 * beta_L * r_energy / (1 + r_energy)
        eta_density_R = 2.0 * beta_R / (1 + r_density)
        eta_momentum_R = 2.0 * beta_R / (1 + r_momentum)
        eta_energy_R = 2.0 * beta_R / (1 + r_energy)

        return eta_density_L, eta_momentum_L, eta_energy_L, eta_density_R, eta_momentum_R, eta_energy_R

    @staticmethod
    def calculate_limited_slopes(left_rho_slope, right_rho_slope,
                                 left_mom_slope, right_mom_slope,
                                 left_energy_slope, right_energy_slope,
                                 c):
        """
        Function that performs an operation to bound the slopes within the TVD region
        """
        raise NotImplementedError("Called from Base Class!")


class MinBeeLimiter(BaseLimiter):
    @staticmethod
    def calculate_limited_slopes(left_rho_slope, right_rho_slope,
                                  left_mom_slope, right_mom_slope,
                                  left_energy_slope, right_energy_slope,
                                  c):
        r_density, r_momentum, r_energy = MinBeeLimiter._calculate_slope_ratios(left_rho_slope, right_rho_slope,
                                                                                left_mom_slope, right_mom_slope,
                                                                                left_energy_slope, right_energy_slope)

        eta_density_L, eta_momentum_L, eta_energy_L, eta_density_R, eta_momentum_R, eta_energy_R = \
            MinBeeLimiter._calculate_eta_values(c,
                                                r_density,
                                                r_momentum,
                                                r_energy)

        eta_density = min(1.0, eta_density_R) if r_density > 1.0 else r_density
        eta_momentum = min(1.0, eta_momentum_R) if r_momentum > 1.0 else r_momentum
        eta_energy = min(1.0, eta_energy_R) if r_energy > 1.0 else r_energy
        if r_density <= 0.0 or r_momentum <= 0.0 or r_energy <= 0.0:
            eta_density = 0.0
            eta_momentum = 0.0
            eta_energy = 0.0

        average_density_slope = 0.5 * (left_rho_slope + right_rho_slope) * eta_density
        average_momentum_slope = 0.5 * (left_mom_slope + right_mom_slope) * eta_momentum
        average_energy_slope = 0.5 * (left_energy_slope + right_energy_slope) * eta_energy

        return average_density_slope, average_momentum_slope, average_energy_slope


class UltraBeeLimiter(BaseLimiter):
    @staticmethod
    def calculate_limited_slopes(left_rho_slope, right_rho_slope,
                                  left_mom_slope, right_mom_slope,
                                  left_energy_slope, right_energy_slope,
                                  c):
        r_density, r_momentum, r_energy = UltraBeeLimiter._calculate_slope_ratios(left_rho_slope, right_rho_slope,
                                                                                left_mom_slope, right_mom_slope,
                                                                                left_energy_slope, right_energy_slope)

        eta_density_L, eta_momentum_L, eta_energy_L, eta_density_R, eta_momentum_R, eta_energy_R = \
            UltraBeeLimiter._calculate_eta_values(c,
                                                r_density,
                                                r_momentum,
                                                r_energy)

        if r_density <= 0.0 or r_momentum <= 0.0 or r_energy <= 0.0:
            eta_density = 0.0
            eta_momentum = 0.0
            eta_energy = 0.0
        else:
            eta_density = min(eta_density_L, eta_density_R)
            eta_momentum = min(eta_momentum_L, eta_momentum_R)
            eta_energy = min(eta_energy_L, eta_energy_R)

        average_density_slope = 0.5 * (left_rho_slope + right_rho_slope) * eta_density
        average_momentum_slope = 0.5 * (left_mom_slope + right_mom_slope) * eta_momentum
        average_energy_slope = 0.5 * (left_energy_slope + right_energy_slope) * eta_energy

        return average_density_slope, average_momentum_slope, average_energy_slope


