"""
Author: Rohan
Date: 29/06/16

This file contains a class used to generate the 1D analytic solution to the shock tube problems using a Riemann solution.
"""

import numpy as np
from matplotlib import pyplot as plt

from CFD_Projects.riemann_solvers.thermodynamic_state import ThermodynamicState
from CFD_Projects.riemann_solvers.riemann_solver import RiemannSolver


class AnalyticShockTube(object):
    def __init__(self, left_state, right_state, membrane_location, num_pts):
        assert isinstance(left_state, ThermodynamicState)
        assert isinstance(right_state, ThermodynamicState)
        assert left_state.gamma == right_state.gamma

        self.left_state = left_state
        self.right_state = right_state
        self.solver = RiemannSolver(left_state.gamma)
        self.membrane_location = membrane_location
        self.x = np.linspace(0, 1, num_pts)
        self.rho = np.zeros(num_pts)
        self.u = np.zeros(num_pts)
        self.p = np.zeros(num_pts)
        self.e = np.zeros(num_pts)

    def __set_rarefaction(self, p_star, u_star, t, is_left):
        assert isinstance(is_left, bool)

        contact_location = self.membrane_location + u_star * t

        gamma = self.left_state.gamma
        gamma_const1 = 2 / (gamma + 1)
        gamma_const2 = (gamma - 1) / (gamma + 1)
        gamma_const3 = (gamma - 1) / 2
        gamma_const4 = 2 * gamma / (gamma - 1)
        if is_left:
            rho = self.left_state.rho
            u = self.left_state.u
            a = self.left_state.a
            p = self.left_state.p
            e = self.left_state.e_int

            #  Get star density state
            rho_star = rho * (p_star / p) ** (1 / gamma)

            # Get high and low rarefaction waves in the envelope
            S_high = u - a
            a_star = np.sqrt(gamma * p_star / rho_star)
            S_low = u_star - a_star

            # Get start and end locations of rarefaction
            x_start = self.membrane_location + S_high * t
            x_end = self.membrane_location + S_low * t

            for i, x in enumerate(self.x):
                if x < x_start:
                    self.rho[i] = rho
                    self.u[i] = u
                    self.p[i] = p
                    self.e[i] = e
                elif x_start <= x < x_end:
                    assert x < self.membrane_location
                    x_left = self.membrane_location - x
                    self.rho[i] = rho * (gamma_const1 - gamma_const2 / a * (-u - x_left / t)) ** (1 / gamma_const3)
                    self.u[i] = gamma_const1 * (a + gamma_const3 * u - x_left / t)
                    self.p[i] = p * (gamma_const1 - gamma_const2 / a * (-u - x_left / t)) ** gamma_const4
                    self.e[i] = self.p[i] / (self.rho[i] * (gamma - 1))
                elif x_end <= x < contact_location:
                    self.rho[i] = rho_star
                    self.u[i] = u_star
                    self.p[i] = p_star
                    self.e[i] = p_star / (rho_star * (gamma - 1))

        else:
            rho = self.right_state.rho
            u = self.right_state.u
            a = self.right_state.a
            p = self.right_state.p

            # Get star density
            rho_star = rho * (p_star / p) ** (1 / gamma)

            # Get high and low wave speeds
            S_high = u + a
            a_star = np.sqrt(gamma * p_star / rho_star)
            S_low = u_star + a_star

            # Get start and end points or rarefaction wave envelope
            x_end = self.membrane_location + S_high * t
            x_start = self.membrane_location + S_low * t

            for i, x in enumerate(self.x):
                if contact_location <= x < x_start:
                    self.rho[i] = rho_star
                    self.u[i] = u_star
                    self.p[i] = p_star
                    self.e[i] = p_star / (rho_star * (gamma - 1))
                elif x_start <= x < x_end:
                    assert self.membrane_location < x
                    x_right = x - self.membrane_location
                    self.rho[i] = rho * (gamma_const1 - gamma_const2 / a * (u - x_right / t)) ** (1 / gamma_const3)
                    self.u[i] = gamma_const1 * (-a + gamma_const3 * u + x_right / t)
                    self.p[i] = p * (gamma_const1 - gamma_const2 / a * (u - x_right / t)) ** gamma_const4
                    self.e[i] = self.p[i] / (self.rho[i] * (gamma - 1))
                elif x >= x_end:
                    self.rho[i] = rho
                    self.u[i] = u
                    self.p[i] = p
                    self.e[i] = p / (rho * (gamma - 1))

    def __set_shock(self, p_star, u_star, t, is_left):
        assert isinstance(is_left, bool)

        # Get contact location
        contact_location = self.membrane_location + u_star * t

        # Get rho star, shock speed and range of shock region
        if is_left:
            rho = self.left_state.rho
            u = self.left_state.u
            a = self.left_state.a
            p = self.left_state.p
            gamma = self.left_state.gamma
            S = u - a * ((gamma + 1) * p_star / (2 * gamma * p) + (gamma - 1) / (2 * gamma)) ** 0.5

            rho_star = rho * ((p_star / p + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * (p_star / p) + 1))

            x_start = self.membrane_location + S * t
            x_end = contact_location

            for i, x in enumerate(self.x):
                if x < x_start:
                    self.rho[i] = rho
                    self.u[i] = u
                    self.p[i] = p
                    self.e[i] = p / (rho * (gamma - 1))
                elif x_start <= x < x_end:
                    self.rho[i] = rho_star
                    self.u[i] = u_star
                    self.p[i] = p_star
                    self.e[i] = p_star / (rho_star * (gamma - 1))
        else:
            rho = self.right_state.rho
            u = self.right_state.u
            a = self.right_state.a
            p = self.right_state.p
            gamma = self.right_state.gamma
            S = u + a * ((gamma + 1) * p_star / (2 * gamma * p) + (gamma - 1) / (2 * gamma)) ** 0.5

            rho_star = rho * ((p_star / p + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * (p_star / p) + 1))

            x_start = contact_location
            x_end = self.membrane_location + S * t

            for i, x in enumerate(self.x):
                if x_start < x <= x_end:
                    self.rho[i] = rho_star
                    self.u[i] = u_star
                    self.p[i] = p_star
                    self.e[i] = p_star / (rho_star * (gamma - 1))
                elif x > x_end:
                    self.rho[i] = rho
                    self.u[i] = u
                    self.p[i] = p
                    self.e[i] = p / (rho * (gamma - 1))

    def __set_left_state(self, p_star, u_star, t):
        if p_star <= self.left_state.p:
            self.__set_rarefaction(p_star, u_star, t, True)
        else:
            self.__set_shock(p_star, u_star, t, True)

    def __set_right_state(self, p_star, u_star, t):
        if p_star <= self.right_state.p:
            self.__set_rarefaction(p_star, u_star, t, False)
        else:
            self.__set_shock(p_star, u_star, t, False)

    def get_solution(self, time):
        """
        Function to get the analytic solution to the Riemann problem at a particular point in time after the removal
        of the membrane.
        :param time: elapsed time after the removal of the membrane.
        :return: Returns arrays for position, pressure, velocity and density at the solution point
        """
        assert isinstance(time, float)

        p_star, u_star = self.solver.get_star_states(self.left_state, self.right_state)

        self.__set_left_state(p_star, u_star, time)
        self.__set_right_state(p_star, u_star, time)

        return self.x, self.rho, self.u, self.p, self.e


def test_sod_problems():
    """
    This function runs through the five shock tube problems outlined in Toro - Chapter 4. The results for contact
    velocity, pressure, and number of iterations should match those on p130-131.
    """
    gamma = 1.4
    p_left = [1.0, 0.4, 1000.0, 0.01, 460.894]
    rho_left = [1.0, 1.0, 1.0, 1.0, 5.99924]
    u_left = [0.0, -2.0, 0.0, 0.0, 19.5975]
    p_right = [0.1, 0.4, 0.01, 100.0, 46.0950]
    rho_right = [0.125, 1.0, 1.0, 1.0, 5.99242]
    u_right = [0.0, 2.0, 0.0, 0.0, -6.19633]
    t = [0.25, 0.15, 0.012, 0.035, 0.035]

    for i in range(0, 5):
        left_state = ThermodynamicState(p_left[i], rho_left[i], u_left[i], gamma)
        right_state = ThermodynamicState(p_right[i], rho_right[i], u_right[i], gamma)

        sod_test = AnalyticShockTube(left_state, right_state, 0.5, 1000)

        x_sol, rho_sol, u_sol, p_sol, e_sol = sod_test.get_solution(t[i])

        title = "Sod Test: {}".format(i + 1)
        num_plts_x = 2
        num_plts_y = 2
        plt.figure(figsize=(10, 10))
        plt.suptitle(title)
        plt.subplot(num_plts_x, num_plts_y, 1)
        plt.title("Density")
        plt.plot(x_sol, rho_sol)
        plt.subplot(num_plts_x, num_plts_y, 2)
        plt.title("Velocity")
        plt.plot(x_sol, u_sol)
        plt.subplot(num_plts_x, num_plts_y, 3)
        plt.title("Pressure")
        plt.plot(x_sol, p_sol)
        plt.subplot(num_plts_x, num_plts_y, 4)
        plt.title("Energy")
        plt.plot(x_sol, e_sol)
        plt.show()


if __name__ == '__main__':
    test_sod_problems()