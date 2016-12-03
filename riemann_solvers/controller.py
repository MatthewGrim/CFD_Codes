"""
Author: Rohan
Date: 01/09/16

This file contains a controller class used to simulate fluid systems.
"""

import numpy as np

from CFD_Projects.riemann_solvers.eos.thermodynamic_state import ThermodynamicState
from CFD_Projects.riemann_solvers.flux_calculator.flux_calculator import FluxCalculator

from CFD_Projects.riemann_solvers.simulations.base_simulation import BaseSimulation1D


class ControllerND(object):
    def _calculate_fluxes(self, dt, ts):
        raise NotImplementedError("Calling from base class!")

    def _calculate_time_step(self):
        raise NotImplementedError("Calling from base class!")

    def _update_states(self, dt):
        raise NotImplementedError("Calling from base class!")

    def _evolve_time_step(self, ts):
        """
        Function carrying out a single timestep
        """
        assert isinstance(ts, int)

        dt = self._calculate_time_step()
        self._calculate_fluxes(dt, ts)
        self._update_states(dt)

        return dt


class Controller1D(ControllerND):
    def __init__(self, simulation):
        assert isinstance(simulation, BaseSimulation1D)
        assert simulation.is_initialised

        # Initialise mesh variables
        self.mesh = simulation.mesh
        self.densities = simulation.densities
        self.pressures = simulation.pressures
        self.velocities = simulation.velocities
        self.internal_energies = simulation.internal_energies
        self.dx = simulation.dx
        self.gamma = simulation.gamma

        # Sim time step variables
        self.final_time = simulation.final_time
        self.CFL = simulation.CFL

        # Initialise flux arrays
        self.density_fluxes = np.zeros(len(self.densities) + 1)
        self.momentum_fluxes = np.zeros(len(self.densities) + 1)
        self.pressure_forces = np.zeros(len(self.densities) + 1)
        self.total_energy_fluxes = np.zeros(len(self.densities) + 1)

        self.flux_calculator = simulation.flux_calculator

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

    def run_sim(self):
        """
        High level simulation function that runs the simulation - effectively the controller
        """
        t = 0.0
        ts = 1
        times = [ t ]
        while t < self.final_time:
            dt = self._evolve_time_step(ts)
            t += dt
            ts += 1
            times.append(t)

        return times, self.mesh, self.densities, self.pressures, self.velocities, self.internal_energies

