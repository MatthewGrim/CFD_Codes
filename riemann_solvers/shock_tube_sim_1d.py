"""
Author: Rohan
Date: 01/09/16

This file contains a class used to simulate a 1D shock tube problem using a Godunov method Riemann solution
"""

import numpy as np

from CFD_Projects.riemann_solvers.thermodynamic_state import ThermodynamicState
from CFD_Projects.riemann_solvers.riemann_solver import RiemannSolver


class ShockTube1D(object):
    def __init__(self, left_state, right_state, membrane_location, final_time, CFL):
        assert(isinstance(left_state, ThermodynamicState))
        assert(isinstance(right_state, ThermodynamicState))
        assert(isinstance(membrane_location, float))
        assert(isinstance(CFL, float))
        assert(isinstance(final_time, double))
        assert(0.0 < membrane_location and 1.0 > membrane_location)
        assert(0.0 < CFL and 1.0 > CFL)

        self.x = np.linspace(0, 1, 1000)
        self.dx = self.x[0] * 2
        self.final_time = final_time
        self.CFL = CFL
        self.solver = RiemannSolver(1.4)

        self.states = list()
        for x_loc in self.x:
            if x_loc < membrane_location:
                self.states.append(left_state)
            else:
                self.states.append(right_state)
        self.density_fluxes = np.zeros(len(self.states) + 1)
        self.momentum_fluxes = np.zeros(len(self.states) + 1)
        self.mass_specific_energy_fluxes = np.zeros(len(self.states) + 1)

    def _calculate_fluxes(self):
        self.density_fluxes = np.zeros(len(self.states) + 1)
        self.momentum_fluxes = np.zeros(len(self.states) + 1)
        self.mass_specific_energy_fluxes = np.zeros(len(self.states) + 1)

        max_wave_speed = 0.0
        for i, state in enumerate(self.states):
            if i == 0 or i == len(self.states) - 1:
                self.solver.get_star_states(self.states[i], self.states[i])
            else:
                self.solver.get_star_states(self.states[i], self.states[i + 1])
            # Store fluxes in array
            # Get wave speed and set max

        return max_wave_speed

    def _calculate_time_step(self, max_wave_speed):
        assert(isinstance(max_wave_speed, float))
        return self.CFL * self.dx / max_wave_speed

    def _update_states(self, dt):
        assert(isinstance(dt, float))

        for i, state in enumerate(self.states):
            new_density = self.states[i].density + self.density_fluxes[i] - self.density_fluxes[i + 1]
            new_momentum = self.states[i].density + self.momentum_fluxes[i] - self.momentum_fluxes[i + 1]
            new_mass_specific_energy = self.states[i].density + self.mass_specific_energy_fluxes[i] - \
                                       self.mass_specific_energy_fluxes[i + 1]

            self.states[i] = ThermodynamicState(new_density, new_momentum, new_mass_specific_energy)

    def _evolve_time_step(self):
        result = list()

        max_wave_speed = self._calculate_fluxes()
        dt = self._calculate_time_step(max_wave_speed)
        self._update_states(dt)

        return dt, self.states

    def run_sim(self):
        t = 0
        result = list()
        times = [ t ]
        result.append(self.states)

        while (t < final_time):
            dt, result_states = self._evolve_time_step()

            t += dt
            times.append(t)
            result.append(result_states)
        return times, result


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

    shock_tube_sim = ShockTube1D(left_state, right_state, membrane_location)

    (times, result) = shock_tube_sim.run_sim()

if __name__ == '__main__':
    example()