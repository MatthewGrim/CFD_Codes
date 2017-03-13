"""
Author: Rohan
Date: 01/09/16

This file contains a controller class used to simulate fluid systems.
"""

import numpy as np

from CFD_Projects.riemann_solvers.eos.thermodynamic_state import ThermodynamicState1D
from CFD_Projects.riemann_solvers.eos.thermodynamic_state import ThermodynamicState2D
from CFD_Projects.riemann_solvers.flux_calculator.flux_calculator import FluxCalculator1D
from CFD_Projects.riemann_solvers.flux_calculator.flux_calculator import FluxCalculator2D
from CFD_Projects.riemann_solvers.boundary_conditions.boundary_condition import BoundaryConditionND

from CFD_Projects.riemann_solvers.simulations.base_simulation import BaseSimulation1D
from CFD_Projects.riemann_solvers.simulations.base_simulation import BaseSimulation2D


class ControllerND(object):
    def _set_boundary_conditions(self):
        raise NotImplementedError("Calling from base class!")

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
        self._set_boundary_conditions()
        self._calculate_fluxes(dt, ts)
        self._update_states(dt)

        return dt


class Controller1D(ControllerND):
    def __init__(self, simulation):
        assert isinstance(simulation, BaseSimulation1D)
        assert simulation.is_initialised

        # Initialise mesh variables
        self.x = simulation.x
        self.densities = simulation.densities
        self.pressures = simulation.pressures
        self.velocities = simulation.vel_x
        self.internal_energies = simulation.internal_energies
        self.dx = simulation.dx
        self.gamma = simulation.gamma

        # Sim time step variables
        self.final_time = simulation.final_time
        self.CFL = simulation.CFL

        # Initialise flux arrays
        self.density_fluxes = np.zeros(len(self.densities) + 1)
        self.momentum_fluxes = np.zeros(len(self.densities) + 1)
        self.total_energy_fluxes = np.zeros(len(self.densities) + 1)

        self.flux_calculator = simulation.flux_calculator
        self.boundary_functions = simulation.boundary_functions

    def _set_boundary_conditions(self):
        """
        Function used to extend the grid at the boundary conditions
        """
        i_length = len(self.densities) - 1
        start_state = ThermodynamicState1D(self.pressures[0], self.densities[0], self.velocities[0], self.gamma)
        end_state = ThermodynamicState1D(self.pressures[i_length], self.densities[i_length], self.velocities[i_length], self.gamma)

        if self.flux_calculator != FluxCalculator1D.MUSCL:
            # Low end
            start_state = self.boundary_functions[BoundaryConditionND.X_LOW](start_state)
            self.densities = np.append([start_state.rho], self.densities, 0)
            self.pressures = np.append([start_state.p], self.pressures, 0)
            self.velocities = np.append([start_state.u], self.velocities, 0)
            self.internal_energies = np.append([start_state.e_int], self.internal_energies, 0)

            # High end
            end_state = self.boundary_functions[BoundaryConditionND.X_HIGH](end_state)
            self.densities = np.append(self.densities, [end_state.rho], 0)
            self.pressures = np.append(self.pressures, [end_state.p], 0)
            self.velocities = np.append(self.velocities, [end_state.u], 0)
            self.internal_energies = np.append(self.internal_energies, [end_state.e_int], 0)
        else:
            # Low end
            start_state = self.boundary_functions[BoundaryConditionND.X_LOW](start_state)
            self.densities = np.append([start_state.rho, start_state.rho], self.densities, 0)
            self.pressures = np.append([start_state.p, start_state.p], self.pressures, 0)
            self.velocities = np.append([start_state.u, start_state.u], self.velocities, 0)
            self.internal_energies = np.append([start_state.e_int, start_state.e_int], self.internal_energies, 0)

            # High end
            end_state = self.boundary_functions[BoundaryConditionND.X_HIGH](end_state)
            self.densities = np.append(self.densities, [end_state.rho, end_state.rho], 0)
            self.pressures = np.append(self.pressures, [end_state.p, end_state.p], 0)
            self.velocities = np.append(self.velocities, [end_state.u, end_state.u], 0)
            self.internal_energies = np.append(self.internal_energies, [end_state.e_int, end_state.e_int], 0)

    def _calculate_fluxes(self, dt, ts):
        """
        Function used to calculate the fluxes between cells using a flux based method
        """
        assert isinstance(ts, int)

        if self.flux_calculator == FluxCalculator1D.GODUNOV:
            self.density_fluxes, \
            self.momentum_fluxes, \
            self.total_energy_fluxes = FluxCalculator1D.calculate_godunov_fluxes(self.densities,
                                                                                 self.pressures,
                                                                                 self.velocities,
                                                                                 self.gamma)
        elif self.flux_calculator == FluxCalculator1D.RANDOM_CHOICE:
            dx_over_dt = self.dx / dt
            self.density_fluxes, \
            self.momentum_fluxes, \
            self.total_energy_fluxes = FluxCalculator1D.calculate_random_choice_fluxes(self.densities,
                                                                                       self.pressures,
                                                                                       self.velocities,
                                                                                       self.gamma,
                                                                                       ts,
                                                                                       dx_over_dt)
        elif self.flux_calculator == FluxCalculator1D.HLLC:
            self.density_fluxes, \
            self.momentum_fluxes, \
            self.total_energy_fluxes = FluxCalculator1D.calculate_hllc_fluxes(self.densities,
                                                                              self.pressures,
                                                                              self.velocities,
                                                                              self.gamma)
        elif self.flux_calculator == FluxCalculator1D.MUSCL:
            self.density_fluxes, \
            self.momentum_fluxes, \
            self.total_energy_fluxes = FluxCalculator1D.calculate_godunov_fluxes(self.densities,
                                                                                 self.pressures,
                                                                                 self.velocities,
                                                                                 self.gamma)
        else:
            raise RuntimeError("Flux calculator does not exist")

        grid_length = len(self.densities)
        if self.flux_calculator != FluxCalculator1D.MUSCL:
            self.densities = self.densities[1 : grid_length - 1]
            self.pressures = self.pressures[1 : grid_length - 1]
            self.velocities = self.velocities[1 : grid_length - 1]
            self.internal_energies = self.internal_energies[1 : grid_length - 1]
        else:
            self.densities = self.densities[2 : grid_length - 2]
            self.pressures = self.pressures[2 : grid_length - 2]
            self.velocities = self.velocities[2 : grid_length - 2]
            self.internal_energies = self.internal_energies[2 : grid_length - 2]

    def _calculate_time_step(self):
        """
        Calculates the time step from the approximation in Toro 6 using max(|u| + a) in the domain
        """
        max_wave_speed = 0.0
        for i, dens in enumerate(self.densities):
            sound_speed = np.sqrt(self.gamma * self.pressures[i] / self.densities[i])
            wave_speed = np.fabs(self.velocities[i]) + sound_speed
            if wave_speed > max_wave_speed:
                max_wave_speed = wave_speed
        if self.flux_calculator == FluxCalculator1D.RANDOM_CHOICE:
            return 0.5 * self.CFL * self.dx / max_wave_speed
        else:
            return self.CFL * self.dx / max_wave_speed

    def _update_states(self, dt):
        """
        Uses the time step and calculated fluxes to update states in each cell
        """
        assert(isinstance(dt, float))

        for i, state in enumerate(self.densities):
            if self.flux_calculator == FluxCalculator1D.RANDOM_CHOICE:
                rho = self.density_fluxes[i]
                momentum = self.momentum_fluxes[i]
                total_energy = self.total_energy_fluxes[i]

                velocity = momentum / rho
                kinetic_energy = 0.5 * rho * velocity ** 2
                internal_energy = total_energy - kinetic_energy
                pressure = internal_energy * (self.gamma - 1)

                state = ThermodynamicState1D(pressure, rho, velocity, self.gamma)
            else:
                total_density_flux = (self.density_fluxes[i] - self.density_fluxes[i + 1]) * dt / self.dx
                total_momentum_flux = (self.momentum_fluxes[i] - self.momentum_fluxes[i + 1]) * dt / self.dx
                total_energy_flux = (self.total_energy_fluxes[i] - self.total_energy_fluxes[i + 1]) * dt / self.dx

                state = ThermodynamicState1D(self.pressures[i], self.densities[i], self.velocities[i], self.gamma)
                state.update_states(total_density_flux, total_momentum_flux, total_energy_flux)

            self.densities[i] = state.rho
            self.pressures[i] = state.p
            self.velocities[i] = state.u
            self.internal_energies[i] = state.e_int

    def run_sim(self):
        """
        High level simulation function that runs the simulation
        """
        t = 0.0
        ts = 1
        times = [ t ]
        while t < self.final_time:
            dt = self._evolve_time_step(ts)
            t += dt
            ts += 1
            times.append(t)
            print("Step " + str(ts) + ": " + str(dt))

        return times, self.x, self.densities, self.pressures, self.velocities, self.internal_energies


class Controller2D(ControllerND):
    def __init__(self, simulation):
        assert isinstance(simulation, BaseSimulation2D)
        assert simulation.is_initialised

        # Initialise mesh variables
        self.x = simulation.x
        self.y = simulation.y
        self.densities = simulation.densities
        self.pressures = simulation.pressures
        self.vel_x = simulation.vel_x
        self.vel_y = simulation.vel_y
        self.internal_energies = simulation.internal_energies
        self.dx = simulation.dx
        self.dy = simulation.dy
        self.delta = min(self.dx, self.dy)
        self.gamma = simulation.gamma

        # Sim time step variables
        self.final_time = simulation.final_time
        self.CFL = simulation.CFL

        # Initialise flux arrays
        self.density_fluxes = np.zeros((np.shape(self.densities)[0] + 1, np.shape(self.densities)[1] + 1, 2))
        self.momentum_fluxes_x = np.zeros((np.shape(self.density_fluxes)))
        self.momentum_fluxes_y = np.zeros((np.shape(self.density_fluxes)))
        self.total_energy_fluxes = np.zeros((np.shape(self.density_fluxes)))

        self.flux_calculator = simulation.flux_calculator
        self.boundary_functions = simulation.boundary_functions

    def _set_boundary_conditions(self):
        """
        Function used to extend the grid at the boundary conditions
        """
        i_length, j_length = np.shape(self.densities)

        # Create new arrays
        new_densities = np.zeros((i_length + 2, j_length + 2))
        new_pressures = np.zeros((i_length + 2, j_length + 2))
        new_vel_x = np.zeros((i_length + 2, j_length + 2))
        new_vel_y = np.zeros((i_length + 2, j_length + 2))
        new_internal_energies = np.zeros((i_length + 2, j_length + 2))
        new_densities[1:-1, 1:-1] = self.densities
        new_pressures[1:-1, 1:-1] = self.pressures
        new_vel_x[1:-1, 1:-1] = self.vel_x
        new_vel_y[1:-1, 1:-1] = self.vel_y
        new_internal_energies[1:-1, 1:-1] = self.internal_energies

        # Handle x boundaries
        for j in range(j_length):
            start_state = ThermodynamicState2D(self.pressures[0, j], self.densities[0, j],
                                               self.vel_x[0, j], self.vel_y[0, j], self.gamma)
            start_state = self.boundary_functions[BoundaryConditionND.X_LOW](start_state, self.y[j])
            new_densities[0, j + 1] = start_state.rho
            new_pressures[0, j + 1] = start_state.p
            new_vel_x[0, j + 1] = start_state.u
            new_vel_y[0, j + 1] = start_state.v
            new_internal_energies[0, j + 1] = start_state.e_int

            end_state = ThermodynamicState2D(self.pressures[-1, j], self.densities[-1, j], self.vel_x[-1, j],
                                             self.vel_y[-1, j], self.gamma)
            end_state = self.boundary_functions[BoundaryConditionND.X_HIGH](end_state, self.y[j])
            new_densities[-1, j + 1] = end_state.rho
            new_pressures[-1, j + 1] = end_state.p
            new_vel_x[-1, j + 1] = end_state.u
            new_vel_y[-1, j + 1] = end_state.v
            new_internal_energies[-1, j + 1] = end_state.e_int

        # Handle y boundaries
        for i in range(i_length + 2):
            if i == 0:
                x = self.x[0]
            elif i == i_length + 1:
                x = self.x[i_length - 1]
            else:
                x = self.x[i - 1]
            start_state = ThermodynamicState2D(new_pressures[i, 1], new_densities[i, 1], new_vel_x[i, 1],
                                               new_vel_y[i, 1], self.gamma)
            start_state = self.boundary_functions[BoundaryConditionND.Y_LOW](start_state, x)
            new_densities[i, 0] = start_state.rho
            new_pressures[i, 0] = start_state.p
            new_vel_x[i, 0] = start_state.u
            new_vel_y[i, 0] = start_state.v
            new_internal_energies[i, 0] = start_state.e_int

            end_state = ThermodynamicState2D(new_pressures[i, -2], new_densities[i, -2], new_vel_x[i, -2],
                                             new_vel_y[i, -2], self.gamma)
            end_state = self.boundary_functions[BoundaryConditionND.Y_HIGH](end_state, x)
            new_densities[i, -1] = end_state.rho
            new_pressures[i, -1] = end_state.p
            new_vel_x[i, -1] = end_state.u
            new_vel_y[i, -1] = end_state.v
            new_internal_energies[i, -1] = end_state.e_int

        # Assign new arrays
        self.densities = new_densities
        self.pressures = new_pressures
        self.vel_x = new_vel_x
        self.vel_y = new_vel_y
        self.internal_energies = new_internal_energies

    def _calculate_fluxes(self, dt, ts):
        """
        Function used to calculate the fluxes between cells using a flux based method
        """
        assert isinstance(ts, int)

        if self.flux_calculator == FluxCalculator2D.GODUNOV:
            self.density_fluxes, \
            self.momentum_fluxes_x, \
            self.momentum_fluxes_y, \
            self.total_energy_fluxes = FluxCalculator2D.calculate_godunov_fluxes(self.densities,
                                                                                 self.pressures,
                                                                                 self.vel_x,
                                                                                 self.vel_y,
                                                                                 self.gamma)
        else:
            raise RuntimeError("Flux calculator does not exist")

        self.densities = self.densities[1:-1, 1:-1]
        self.pressures = self.pressures[1:-1, 1:-1]
        self.vel_x = self.vel_x[1:-1, 1:-1]
        self.vel_y = self.vel_y[1:-1, 1:-1]
        self.internal_energies = self.internal_energies[1:-1, 1:-1]

    def _calculate_time_step(self):
        """
        Calculates the time step from the approximation in Toro 6 using max(|u| + a) in the domain
        """
        max_wave_speed = 0.0
        i_length, j_length = np.shape(self.densities)
        for i in range(i_length):
            for j in range(j_length):
                sound_speed = np.sqrt(self.gamma * self.pressures[i, j] / self.densities[i, j])
                wave_speed_1 = np.fabs(self.vel_x[i, j]) + sound_speed
                wave_speed_2 = np.fabs(self.vel_y[i, j]) + sound_speed
                wave_speed = max(wave_speed_1, wave_speed_2)
                if wave_speed > max_wave_speed:
                    max_wave_speed = wave_speed
        return self.CFL * self.delta / max_wave_speed

    def _update_states(self, dt):
        """
        Uses the time step and calculated fluxes to update states in each cell
        """
        assert(isinstance(dt, float))
        (i_length, j_length) = np.shape(self.densities)

        total_dens_flux = 0.0
        total_mom_flux_x = 0.0
        total_mom_flux_y = 0.0
        total_e_flux = 0.0
        for i in range(i_length):
            for j in range(j_length):
                total_density_flux = (self.density_fluxes[i, j, 0] - self.density_fluxes[i + 1, j, 0]) * dt / self.dx + \
                                         (self.density_fluxes[i, j, 1] - self.density_fluxes[i, j + 1, 1]) * dt / self.dy
                total_momentum_flux_x = (self.momentum_fluxes_x[i, j, 0] - self.momentum_fluxes_x[i + 1, j, 0]) * dt / self.dx + \
                                        (self.momentum_fluxes_x[i, j, 1] - self.momentum_fluxes_x[i, j + 1, 1]) * dt / self.dy
                total_momentum_flux_y = (self.momentum_fluxes_y[i, j, 0] - self.momentum_fluxes_y[i + 1, j, 0]) * dt / self.dx + \
                                        (self.momentum_fluxes_y[i, j, 1] - self.momentum_fluxes_y[i, j + 1, 1]) * dt / self.dy
                total_energy_flux = (self.total_energy_fluxes[i, j, 0] - self.total_energy_fluxes[i + 1, j, 0]) * dt / self.dx + \
                                    (self.total_energy_fluxes[i, j, 1] - self.total_energy_fluxes[i, j + 1, 1]) * dt / self.dy

                state = ThermodynamicState2D(self.pressures[i, j], self.densities[i, j],
                                             self.vel_x[i, j], self.vel_y[i, j], self.gamma)

                state.update_states(total_density_flux, total_momentum_flux_x, total_momentum_flux_y,
                                    total_energy_flux)

                self.densities[i, j] = state.rho
                self.pressures[i, j] = state.p
                self.vel_x[i, j] = state.u
                self.vel_y[i, j] = state.v
                self.internal_energies[i, j] = state.e_int

                total_dens_flux += total_density_flux
                total_mom_flux_x += total_momentum_flux_x
                total_mom_flux_y += total_momentum_flux_y
                total_e_flux += total_energy_flux

    def run_sim(self):
        """
        High level simulation function that runs the simulation
        """
        t = 0.0
        ts = 1
        times = [ t ]
        while t < self.final_time:
            dt = self._evolve_time_step(ts)
            t += dt
            times.append(t)
            print("Step " + str(ts) + ": " + str(t))
            ts += 1

        return times, self.x, self.y, self.densities, self.pressures, self.vel_x, self.vel_y, self.internal_energies