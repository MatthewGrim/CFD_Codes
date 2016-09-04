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

        self.x = np.linspace(0, 1, 100)
        self.dx = self.x[1] * 2
        self.final_time = final_time
        self.CFL = CFL
        self.solver = RiemannSolver(1.4)

        self.densities = list()
        self.pressures = list()
        self.velocities = list()
        self.gamma = left_state.gamma
        for x_loc in self.x:
            if x_loc < membrane_location:
                self.densities.append(left_state.rho)
                self.pressures.append(left_state.p)
                self.velocities.append(left_state.u)
            else:
                self.densities.append(right_state.rho)
                self.pressures.append(right_state.p)
                self.velocities.append(right_state.u)
        self.densities = np.asarray(self.densities)
        self.pressures = np.asarray(self.pressures)
        self.velocities = np.asarray(self.velocities)

        self.density_fluxes = np.zeros(len(self.densities) + 1)
        self.momentum_fluxes = np.zeros(len(self.densities) + 1)
        self.mass_specific_energy_fluxes = np.zeros(len(self.densities) + 1)

    def _calculate_fluxes(self):
        self.density_fluxes = np.zeros(len(self.densities) + 1)
        self.momentum_fluxes = np.zeros(len(self.densities) + 1)
        self.mass_specific_energy_fluxes = np.zeros(len(self.densities) + 1)

        max_wave_speed = 0.0
        for i, density in enumerate(self.densities):
            if i == len(self.densities) - 1:
                continue

            left_state = ThermodynamicState(self.pressures[i], self.densities[i], self.velocities[i], self.gamma)
            right_state = ThermodynamicState(self.pressures[i + 1], self.densities[i + 1], self.velocities[i + 1], self.gamma)

            p_star, u_star = self.solver.get_star_states(left_state, right_state)

            rho = left_state.rho
            p = left_state.p
            gamma = left_state.gamma
            if left_state.p > p_star:
                rho_star = rho * ((p_star / p + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * (p_star / p) + 1))
            else:
                rho_star = rho * (p_star / p) ** (1 / gamma)

            # Store fluxes in array
            # print "Rho flux: " + str(rho_star * u_star)

            self.density_fluxes[i + 1] = rho_star * u_star
            self.momentum_fluxes[i + 1] = rho_star * u_star * u_star
            self.mass_specific_energy_fluxes[i + 1] = rho_star * u_star * u_star * u_star

            # Get wave speed and set max
            if left_state.p > p_star:
                wave_left = left_state.u - left_state.a * ((gamma + 1) * p_star / (2 * gamma * left_state.p) + (gamma - 1) / (2 * gamma)) ** 0.5
            else:
                wave_left = left_state.u - left_state.a

            if right_state.p > p_star:
                wave_right = left_state.u + left_state.a * ((gamma + 1) * p_star / (2 * gamma * left_state.p) + (gamma - 1) / (2 * gamma)) ** 0.5
            else:
                wave_right = left_state.u + left_state.a

            wave_speed = max(wave_left, wave_right)

            if wave_speed > max_wave_speed:
                max_wave_speed = wave_speed

        return max_wave_speed

    def _calculate_time_step(self, max_wave_speed):
        assert(isinstance(max_wave_speed, float))
        return self.CFL * self.dx / max_wave_speed

    def _update_states(self, dt):
        assert(isinstance(dt, float))

        for i, state in enumerate(self.densities):
            total_density_flux = (self.density_fluxes[i] - self.density_fluxes[i + 1]) * dt / self.dx
            total_momentum_flux = (self.momentum_fluxes[i] - self.momentum_fluxes[i + 1]) * dt / self.dx
            total_energy_flux = (self.mass_specific_energy_fluxes[i] - self.mass_specific_energy_fluxes[i + 1]) * dt / self.dx

            state = ThermodynamicState(self.pressures[i], self.densities[i], self.velocities[i], self.gamma)

            state.update_states(total_density_flux, total_momentum_flux, total_energy_flux)

            self.densities[i] = state.rho
            self.pressures[i] = state.p
            self.velocities[i] = state.u

    def _evolve_time_step(self):
        result = list()

        # print self.densities

        # print self.density_fluxes

        max_wave_speed = self._calculate_fluxes()

        # print self.density_fluxes

        dt = self._calculate_time_step(max_wave_speed)
        self._update_states(dt)

        print self.pressures

        return dt

    def run_sim(self):
        t = 0

        times = [ t ]

        i = 0
        while i < 5:
        # while t < self.final_time:
            dt = self._evolve_time_step()

            t += dt
            times.append(t)

            plt.figure()
            plt.plot(self.x, self.densities)
            plt.show()

            i += 1


        return times, self.x, self.densities, self.pressures, self.velocities


def example():
    gamma = 1.4
    p_left = 1.0
    rho_left = 1.0
    u_left = 0.0
    p_right = 0.1
    rho_right = 0.125
    u_right = 0.0
    membrane_location = 0.5

    left_state = ThermodynamicState(p_left, rho_left, u_left, gamma)
    right_state = ThermodynamicState(p_right, rho_right, u_right, gamma)

    shock_tube_sim = ShockTube1D(left_state, right_state, membrane_location, final_time=0.25, CFL=0.9)

    (times, x, densities, pressures, velocities) = shock_tube_sim.run_sim()
    # assert(len(times) == len(densities))

    print times[-1]
    plt.figure()
    plt.plot(x, densities)
    plt.show()

if __name__ == '__main__':
    example()