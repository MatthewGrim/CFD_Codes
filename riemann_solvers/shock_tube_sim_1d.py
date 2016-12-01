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
from CFD_Projects.riemann_solvers.analytic_shock_tube import AnalyticShockTube
from CFD_Projects.riemann_solvers.flux_calculator import FluxCalculator


class ShockTube1D(object):
    def __init__(self, left_state, right_state, membrane_location, final_time, CFL, flux_calculator):
        assert(isinstance(left_state, ThermodynamicState))
        assert(isinstance(right_state, ThermodynamicState))
        assert(isinstance(membrane_location, float))
        assert(isinstance(CFL, float))
        assert(isinstance(final_time, float))
        assert(0.0 < membrane_location < 1.0)
        assert(0.0 < CFL < 1.0)

        self.x = np.linspace(0.005, 0.995, 100)
        self.dx = self.x[0] * 2
        self.final_time = final_time
        self.CFL = CFL
        self.solver = RiemannSolver(left_state.gamma)
        self.flux_calculator = flux_calculator

        # Initialise physical states
        self.densities = list()
        self.pressures = list()
        self.velocities = list()
        self.internal_energies = list()
        self.gamma = left_state.gamma
        for x_loc in self.x:
            if x_loc < membrane_location:
                self.densities.append(left_state.rho)
                self.pressures.append(left_state.p)
                self.velocities.append(left_state.u)
                self.internal_energies.append(left_state.e_int)
            else:
                self.densities.append(right_state.rho)
                self.pressures.append(right_state.p)
                self.velocities.append(right_state.u)
                self.internal_energies.append(right_state.e_int)
        self.densities = np.asarray(self.densities)
        self.pressures = np.asarray(self.pressures)
        self.velocities = np.asarray(self.velocities)
        self.internal_energies = np.asarray(self.internal_energies)

        # Initialise flux arrays
        self.density_fluxes = np.zeros(len(self.densities) + 1)
        self.momentum_fluxes = np.zeros(len(self.densities) + 1)
        self.pressure_forces = np.zeros(len(self.densities) + 1)
        self.total_energy_fluxes = np.zeros(len(self.densities) + 1)

    def _calculate_fluxes(self, dt, ts):
        """
        Function used to calculate the fluxes between cells using a flux based method
        """
        assert isinstance(ts, int)

        if self.flux_calculator == FluxCalculator.GODUNOV:
            self.density_fluxes, \
            self.momentum_fluxes, \
            self.total_energy_fluxes = FluxCalculator.calculate_godunov_fluxes(self.densities,
                                                                               self.pressures,
                                                                               self.velocities,
                                                                               self.gamma)
        elif self.flux_calculator == FluxCalculator.RANDOM_CHOICE:
            dx_over_dt = self.dx / dt
            self.density_fluxes, \
            self.momentum_fluxes, \
            self.total_energy_fluxes = FluxCalculator.calculate_random_choice_fluxes(self.densities,
                                                                                     self.pressures,
                                                                                     self.velocities,
                                                                                     self.gamma,
                                                                                     ts,
                                                                                     dx_over_dt)
        else:
            raise RuntimeError("Flux calculator does not exist")

    def _calculate_time_step(self):
        """
        Calculates the time step from the approximation in Toro 6 using max(|u| + a) in the domain
        """
        max_wave_speed = 0.0
        for i, dens in enumerate(self.densities):
            state = ThermodynamicState(self.pressures[i], self.densities[i], self.velocities[i], self.gamma)
            wave_speed = np.fabs(state.u) + state.sound_speed()
            if wave_speed > max_wave_speed:
                max_wave_speed = wave_speed
        if self.flux_calculator == FluxCalculator.RANDOM_CHOICE:
            return 0.5 * self.CFL * self.dx / max_wave_speed
        else:
            return self.CFL * self.dx / max_wave_speed

    def _update_states(self, dt):
        """
        Uses the time step and calculated fluxes to update states in each cell
        """
        assert(isinstance(dt, float))

        for i, state in enumerate(self.densities):
            if self.flux_calculator == FluxCalculator.RANDOM_CHOICE:
                rho = self.density_fluxes[i]
                momentum = self.momentum_fluxes[i]
                total_energy = self.total_energy_fluxes[i]

                velocity = momentum / rho
                kinetic_energy = 0.5 * rho * velocity ** 2
                internal_energy = total_energy - kinetic_energy
                pressure = internal_energy * (self.gamma - 1)

                state = ThermodynamicState(pressure, rho, velocity, self.gamma)
            else:
                total_density_flux = (self.density_fluxes[i] - self.density_fluxes[i + 1]) * dt / self.dx
                total_momentum_flux = (self.momentum_fluxes[i] - self.momentum_fluxes[i + 1]) * dt / self.dx
                total_energy_flux = (self.total_energy_fluxes[i] - self.total_energy_fluxes[i + 1]) * dt / self.dx

                state = ThermodynamicState(self.pressures[i], self.densities[i], self.velocities[i], self.gamma)
                state.update_states(total_density_flux, total_momentum_flux, total_energy_flux, i)

            self.densities[i] = state.rho
            self.pressures[i] = state.p
            self.velocities[i] = state.u
            self.internal_energies[i] = state.e_int

    def _evolve_time_step(self, ts):
        """
        Function carrying out a single timestep
        """
        assert isinstance(ts, int)

        dt = self._calculate_time_step()
        self._calculate_fluxes(dt, ts)
        self._update_states(dt)

        return dt

    def run_sim(self):
        """
        High level simulation function that runs the simulation - effectively the controller
        """
        t = 0
        ts = 1
        times = [ t ]
        while t < self.final_time:
            dt = self._evolve_time_step(ts)
            t += dt
            ts += 1
            times.append(t)

        return times, self.x, self.densities, self.pressures, self.velocities, self.internal_energies


def example():
    """
    Runs the problems from Toro Chapter 6 to validate this simulation
    """
    gamma = 1.4
    p_left = [1.0, 0.4, 1000.0, 460.894, 1000.0]
    rho_left = [1.0, 1.0, 1.0, 5.99924, 1.0]
    u_left = [0.75, -2.0, 0.0, 19.5975, -19.5975]
    p_right = [0.1, 0.4, 0.01, 46.0950, 0.01]
    rho_right = [0.125, 1.0, 1.0, 5.99242, 1.0]
    u_right = [0.0, 2.0, 0.0, -6.19633, -19.59745]
    membrane_location = [0.3, 0.5, 0.5, 0.4, 0.8]
    end_times = [0.25, 0.15, 0.012, 0.035, 0.012]

    for i in range(0, 5):
        left_state = ThermodynamicState(p_left[i], rho_left[i], u_left[i], gamma)
        right_state = ThermodynamicState(p_right[i], rho_right[i], u_right[i], gamma)

        shock_tube_sim_god = ShockTube1D(left_state, right_state, membrane_location[i],
                                     final_time=end_times[i], CFL=0.9,
                                     flux_calculator=FluxCalculator.GODUNOV)
        shock_tube_sim_rc = ShockTube1D(left_state, right_state, membrane_location[i],
                                     final_time=end_times[i], CFL=0.9,
                                     flux_calculator=FluxCalculator.RANDOM_CHOICE)

        (times_god, x_god, densities_god, pressures_god, velocities_god, internal_energies_god) = shock_tube_sim_god.run_sim()
        (times_rc, x_rc, densities_rc, pressures_rc, velocities_rc, internal_energies_rc) = shock_tube_sim_rc.run_sim()

        sod_test = AnalyticShockTube(left_state, right_state, membrane_location[i], 1000)

        x_sol, rho_sol, u_sol, p_sol, e_sol = sod_test.get_solution(times_god[-1], membrane_location[i])

        title = "Sod Test: {}".format(i + 1)
        num_plts_x = 2
        num_plts_y = 2
        plt.figure(figsize=(20, 10))
        plt.suptitle(title)
        plt.subplot(num_plts_x, num_plts_y, 1)
        plt.title("Density")
        plt.plot(x_sol, rho_sol)
        plt.scatter(x_god, densities_god, c='g')
        plt.scatter(x_rc, densities_rc, c='r')
        plt.xlim([0.0, 1.0])
        plt.subplot(num_plts_x, num_plts_y, 2)
        plt.title("Velocity")
        plt.plot(x_sol, u_sol)
        plt.scatter(x_god, velocities_god, c='g')
        plt.scatter(x_rc, velocities_rc, c='r')
        plt.xlim([0.0, 1.0])
        plt.subplot(num_plts_x, num_plts_y, 3)
        plt.title("Pressure")
        plt.plot(x_sol, p_sol)
        plt.scatter(x_god, pressures_god, c='g')
        plt.scatter(x_rc, pressures_rc, c='r')
        plt.xlim([0.0, 1.0])
        plt.subplot(num_plts_x, num_plts_y, 4)
        plt.title("Internal Energy")
        plt.plot(x_sol, e_sol)
        plt.scatter(x_god, internal_energies_god, c='g')
        plt.scatter(x_rc, internal_energies_rc, c='r')
        plt.xlim([0.0, 1.0])
        plt.show()


if __name__ == '__main__':
    example()