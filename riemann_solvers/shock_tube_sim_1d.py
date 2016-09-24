"""
Author: Rohan
Date: 01/09/16

This file contains a class used to simulate a 1D shock tube problem using a Godunov method Riemann solution. The code in
this file should replicate the results from Toro - Chapter 6
"""

import numpy as np
from matplotlib import pyplot as plt

from CFD_Projects.riemann_solvers.thermodynamic_state import ThermodynamicState
from CFD_Projects.riemann_solvers.riemann_solver import RiemannSolver


class ShockTube1D(object):
    def __init__(self, left_state, right_state, membrane_location, final_time, CFL):
        assert(isinstance(left_state, ThermodynamicState))
        assert(isinstance(right_state, ThermodynamicState))
        assert(isinstance(membrane_location, float))
        assert(isinstance(CFL, float))
        assert(isinstance(final_time, float))
        assert(0.0 < membrane_location and 1.0 > membrane_location)
        assert(0.0 < CFL and 1.0 > CFL)

        self.x = np.linspace(0.005, 0.995, 100)
        self.dx = self.x[0] * 2
        self.final_time = final_time
        self.CFL = CFL
        self.solver = RiemannSolver(left_state.gamma)

        self.densities = list()
        self.pressures = list()
        self.velocities = list()
        self.internal_energies = list()
        self.gamma = left_state.gamma
        total_mass = 0.0
        for x_loc in self.x:
            if x_loc < membrane_location:
                self.densities.append(left_state.rho)
                self.pressures.append(left_state.p)
                self.velocities.append(left_state.u)
                self.internal_energies.append(left_state.e_int)
                total_mass += left_state.rho * self.dx
            else:
                self.densities.append(right_state.rho)
                self.pressures.append(right_state.p)
                self.velocities.append(right_state.u)
                self.internal_energies.append(right_state.e_int)
                total_mass += right_state.rho * self.dx
        self.densities = np.asarray(self.densities)
        self.pressures = np.asarray(self.pressures)
        self.velocities = np.asarray(self.velocities)
        self.internal_energies = np.asarray(self.internal_energies)
        self.mass_conservation = total_mass

        self.density_fluxes = np.zeros(len(self.densities) + 1)
        self.momentum_fluxes = np.zeros(len(self.densities) + 1)
        self.total_energy_fluxes = np.zeros(len(self.densities) + 1)

    def _calculate_fluxes(self):
        """
        Function used to calculate the fluxes between cells as well as the maximum wave speed in the simulation domain
        for the CFL condition
        """
        self.density_fluxes = np.zeros(len(self.densities) + 1)
        self.momentum_fluxes = np.zeros(len(self.densities) + 1)
        self.total_energy_fluxes = np.zeros(len(self.densities) + 1)

        max_wave_speed = 0.0
        for i, dens_flux in enumerate(self.density_fluxes):
            # Generate left and right states from cell averaged values
            if i == 0:
                left_state = ThermodynamicState(self.pressures[i], self.densities[i], self.velocities[i], self.gamma)
                right_state = left_state
            elif i == len(self.density_fluxes) - 1:
                left_state = ThermodynamicState(self.pressures[i - 1], self.densities[i - 1], self.velocities[i - 1], self.gamma)
                right_state = left_state
            else:
                left_state = ThermodynamicState(self.pressures[i - 1], self.densities[i - 1], self.velocities[i - 1], self.gamma)
                right_state = ThermodynamicState(self.pressures[i], self.densities[i], self.velocities[i], self.gamma)

            # Solve Riemann problem for star states
            p_star, u_star = self.solver.get_star_states(left_state, right_state)

            # Find state at boundary
            if u_star < 0:
                # Consider right wave structures
                rho = right_state.rho
                p = right_state.p
                gamma = right_state.gamma
                if right_state.p >= p_star:
                    rho_star = rho * (p_star / p) ** (1 / gamma)
                    a_star = np.sqrt(gamma * p_star / rho_star)
                    wave_right_high = right_state.u + right_state.a
                    wave_right_low = u_star + a_star
                    if wave_right_high < 0.0:
                        p_flux = right_state.p
                        rho_flux = right_state.rho
                        u_flux = right_state.u
                    else:
                        if wave_right_low > 0.0:
                            p_flux = p_star
                            rho_flux = rho_star
                            u_flux = u_star
                        else:
                            multiplier = ((2.0 / (gamma + 1)) - (gamma - 1) * right_state.u / (right_state.a * (gamma + 1))) ** (2.0 / (gamma - 1.0))
                            rho_flux = right_state.rho * multiplier
                            u_flux = (2.0 / (gamma + 1)) * (-right_state.a + (gamma - 1) * right_state.u / 2.0)
                            p_flux = right_state.p * multiplier
                else:
                    rho_star = rho * ((p_star / p + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * (p_star / p) + 1))
                    wave_right_shock = right_state.u + right_state.a * ((gamma + 1) * p_star / (2 * gamma * right_state.p) + (gamma - 1) / (2 * gamma)) ** 0.5
                    if wave_right_shock < 0.0:
                        p_flux = right_state.p
                        rho_flux = right_state.rho
                        u_flux = right_state.u
                    else:
                        p_flux = p_star
                        rho_flux = rho_star
                        u_flux = u_star
            else:
                # Consider left wave structures
                rho = left_state.rho
                p = left_state.p
                gamma = left_state.gamma
                if left_state.p >= p_star:
                    rho_star = rho * (p_star / p) ** (1 / gamma)
                    a_star = np.sqrt(gamma * p_star / rho_star)
                    wave_left_high = left_state.u - left_state.a
                    wave_left_low = u_star - a_star
                    if wave_left_high > 0.0:
                        p_flux = left_state.p
                        u_flux = left_state.u
                        rho_flux = left_state.rho
                    else:
                        if wave_left_low < 0.0:
                            p_flux = p_star
                            u_flux = u_star
                            rho_flux = rho_star
                        else:
                            multiplier = ((2.0 / (gamma + 1)) + (gamma - 1) * left_state.u / (left_state.a * (gamma + 1))) ** (2.0 / (gamma - 1.0))
                            rho_flux = left_state.rho * multiplier
                            u_flux = (2.0 / (gamma + 1)) * (left_state.a + (gamma - 1) * left_state.u / 2.0)
                            p_flux = left_state.p * multiplier
                else:
                    rho_star = rho * ((p_star / p + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * (p_star / p) + 1))
                    wave_left_shock = left_state.u - left_state.a * ((gamma + 1) * p_star / (2 * gamma * left_state.p) + (gamma - 1) / (2 * gamma)) ** 0.5
                    if wave_left_shock > 0.0:
                        p_flux = left_state.p
                        u_flux = left_state.u
                        rho_flux = left_state.rho
                    else:
                        p_flux = p_star
                        u_flux = u_star
                        rho_flux = rho_star

            # Store fluxes in array
            self.density_fluxes[i] = rho_flux * u_flux
            self.momentum_fluxes[i] = rho_flux * u_flux * u_flux * np.sign(u_flux) + p_flux
            e_tot = p_flux / (gamma - 1) + 0.5 * rho_flux * u_flux * u_flux
            self.total_energy_fluxes[i] = (p_flux + e_tot) * u_flux

    def _calculate_time_step(self):
        max_wave_speed = 0.0
        for i, dens in enumerate(self.densities):
            state = ThermodynamicState(self.pressures[i], self.densities[i], self.velocities[i], self.gamma)
            wave_speed = np.abs(state.u) + state.a
            if (wave_speed > max_wave_speed):
                max_wave_speed = wave_speed

        # print max_wave_speed
        return self.CFL * self.dx / max_wave_speed

    def _update_states(self, dt):
        assert(isinstance(dt, float))

        for i, state in enumerate(self.densities):
            total_density_flux = (self.density_fluxes[i] - self.density_fluxes[i + 1]) * dt / self.dx
            total_momentum_flux = (self.momentum_fluxes[i] - self.momentum_fluxes[i + 1]) * dt / self.dx
            total_energy_flux = (self.total_energy_fluxes[i] - self.total_energy_fluxes[i + 1]) * dt / self.dx

            state = ThermodynamicState(self.pressures[i], self.densities[i], self.velocities[i], self.gamma)

            state.update_states(total_density_flux, total_momentum_flux, total_energy_flux)

            self.densities[i] = state.rho
            self.pressures[i] = state.p
            self.velocities[i] = state.u
            self.internal_energies[i] = state.e_int

    def _evolve_time_step(self):
        self._calculate_fluxes()

        dt = self._calculate_time_step()

        self._update_states(dt)

        return dt

    def run_sim(self):
        t = 0

        times = [ t ]

        while t < self.final_time:
            dt = self._evolve_time_step()

            # print dt

            t += dt
            times.append(t)

            # title = "Sod Test: {}".format(1)
            # num_plts_x = 2
            # num_plts_y = 2
            # plt.figure(figsize=(10, 10))
            # plt.suptitle(title)
            # plt.subplot(num_plts_x, num_plts_y, 1)
            # plt.title("Density")
            # plt.plot(self.x, self.densities)
            # plt.subplot(num_plts_x, num_plts_y, 2)
            # plt.title("Velocity")
            # plt.plot(self.x, self.velocities)
            # plt.subplot(num_plts_x, num_plts_y, 3)
            # plt.title("Pressure")
            # plt.plot(self.x, self.pressures)
            # plt.subplot(num_plts_x, num_plts_y, 4)
            # plt.title("Internal Energy")
            # plt.plot(self.x, self.internal_energies)
            # plt.show()

        return times, self.x, self.densities, self.pressures, self.velocities, self.internal_energies


def example():
    gamma = 1.4
    p_left = [1.0, 0.4, 1000.0, 460.894, 1000.0]
    rho_left = [1.0, 1.0, 1.0, 5.99924, 1.0]
    u_left = [0.75, -2.0, 0.0, 19.5975, -19.5975]
    p_right = [0.1, 0.4, 0.01, 46.0950, 0.01]
    rho_right = [0.125, 1.0, 1.0, 5.99242, 1.0]
    u_right = [0.0, 2.0, 0.0, -6.19633, -19.59745]
    membrane_location = [0.5, 0.5, 0.5, 0.4, 0.8]
    end_times = [0.25, 0.15, 0.012, 0.035, 0.012]

    for i in range(0, 5):
        left_state = ThermodynamicState(p_left[i], rho_left[i], u_left[i], gamma)
        right_state = ThermodynamicState(p_right[i], rho_right[i], u_right[i], gamma)

        shock_tube_sim = ShockTube1D(left_state, right_state, membrane_location[i], final_time=end_times[i], CFL=0.9)

        (times, x, densities, pressures, velocities, internal_energies) = shock_tube_sim.run_sim()

        # print "Initial Mass: " + str(shock_tube_sim.mass_conservation)

        # print times[-1]
        title = "Sod Test: {}".format(1)
        num_plts_x = 2
        num_plts_y = 2
        plt.figure(figsize=(10, 10))
        plt.suptitle(title)
        plt.subplot(num_plts_x, num_plts_y, 1)
        plt.title("Density")
        plt.plot(x, densities)
        plt.subplot(num_plts_x, num_plts_y, 2)
        plt.title("Velocity")
        plt.plot(x, velocities)
        plt.subplot(num_plts_x, num_plts_y, 3)
        plt.title("Pressure")
        plt.plot(x, pressures)
        plt.subplot(num_plts_x, num_plts_y, 4)
        plt.title("Internal Energy")
        plt.plot(x, internal_energies)
        plt.show()


if __name__ == '__main__':
    example()