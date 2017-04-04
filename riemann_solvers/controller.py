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

        # Initialise multi-phase variables
        self.number_of_fluids = simulation.number_of_fluids
        self.molar_masses = simulation.molar_masses
        self.mass_ratios = simulation.mass_ratios

        # Sim time step variables
        self.final_time = simulation.final_time
        self.CFL = simulation.CFL

        # Initialise flux arrays
        self.density_fluxes = np.zeros((len(self.densities) + 1, self.number_of_fluids))
        self.momentum_fluxes = np.zeros(len(self.densities) + 1)
        self.total_energy_fluxes = np.zeros(len(self.densities) + 1)

        self.flux_calculator = simulation.flux_calculator
        self.boundary_functions = simulation.boundary_functions

        self.validate()

    def validate(self):
        """
        Check that variables passed to Controller by simulation are the correct types and sizes
        """
        assert isinstance(self.x, np.ndarray) and self.x.ndim == 1
        assert isinstance(self.densities, np.ndarray) and self.densities.ndim == 1
        assert isinstance(self.velocities, np.ndarray) and self.velocities.ndim == 1
        assert isinstance(self.pressures, np.ndarray) and self.pressures.ndim == 1
        assert isinstance(self.internal_energies, np.ndarray) and self.internal_energies.ndim == 1
        assert isinstance(self.gamma, np.ndarray) and self.gamma.ndim == 1

        assert isinstance(self.number_of_fluids, int)
        assert isinstance(self.molar_masses, np.ndarray)
        assert self.molar_masses.shape[0] == self.number_of_fluids
        assert self.mass_ratios.shape == (self.x.shape[0], self.number_of_fluids)

        if self.number_of_fluids > 1:
            assert self.flux_calculator is not FluxCalculator1D.RANDOM_CHOICE, "Method is currently incompatible with multi-material flow"
            assert self.flux_calculator is not FluxCalculator1D.HLLC, "Method if currently incompatible with multi material flow"

        assert self.x.shape == self.densities.shape == self.velocities.shape
        assert self.velocities.shape == self.pressures.shape == self.internal_energies.shape

        assert isinstance(self.dx, float)
        assert isinstance(self.CFL, float)
        assert isinstance(self.final_time, float)

    def _set_boundary_conditions(self):
        """
        Function used to extend the grid at the boundary conditions
        """
        i_length = len(self.densities) - 1
        start_state = ThermodynamicState1D(self.pressures[0], self.densities[0], self.velocities[0], self.gamma[0])
        end_state = ThermodynamicState1D(self.pressures[i_length], self.densities[i_length],
                                         self.velocities[i_length], self.gamma[i_length])

        if self.flux_calculator != FluxCalculator1D.MUSCL:
            new_mass_ratios = np.zeros((i_length + 3, self.number_of_fluids))
            new_mass_ratios[1:-1, :] = self.mass_ratios
            # Low end
            start_state = self.boundary_functions[BoundaryConditionND.X_LOW](start_state)
            self.densities = np.append([start_state.rho], self.densities, 0)
            self.pressures = np.append([start_state.p], self.pressures, 0)
            self.velocities = np.append([start_state.u], self.velocities, 0)
            self.internal_energies = np.append([start_state.e_int], self.internal_energies, 0)
            self.gamma = np.append([start_state.gamma], self.gamma, 0)
            new_mass_ratios[0, :] = start_state.mass_ratios

            # High end
            end_state = self.boundary_functions[BoundaryConditionND.X_HIGH](end_state)
            self.densities = np.append(self.densities, [end_state.rho], 0)
            self.pressures = np.append(self.pressures, [end_state.p], 0)
            self.velocities = np.append(self.velocities, [end_state.u], 0)
            self.internal_energies = np.append(self.internal_energies, [end_state.e_int], 0)
            self.gamma = np.append(self.gamma, [end_state.gamma], 0)
            new_mass_ratios[-1, :] = start_state.mass_ratios
        else:
            new_mass_ratios = np.zeros((i_length + 5, self.number_of_fluids))
            new_mass_ratios[2:-2, :] = self.mass_ratios
            # Low end
            start_state = self.boundary_functions[BoundaryConditionND.X_LOW](start_state)
            self.densities = np.append([start_state.rho, start_state.rho], self.densities, 0)
            self.pressures = np.append([start_state.p, start_state.p], self.pressures, 0)
            self.velocities = np.append([start_state.u, start_state.u], self.velocities, 0)
            self.internal_energies = np.append([start_state.e_int, start_state.e_int], self.internal_energies, 0)
            self.gamma = np.append([start_state.gamma, start_state.gamma], self.gamma, 0)
            new_mass_ratios[0, :] = start_state.mass_ratios
            new_mass_ratios[1, :] = start_state.mass_ratios

            # High end
            end_state = self.boundary_functions[BoundaryConditionND.X_HIGH](end_state)
            self.densities = np.append(self.densities, [end_state.rho, end_state.rho], 0)
            self.pressures = np.append(self.pressures, [end_state.p, end_state.p], 0)
            self.velocities = np.append(self.velocities, [end_state.u, end_state.u], 0)
            self.internal_energies = np.append(self.internal_energies, [end_state.e_int, end_state.e_int], 0)
            self.gamma = np.append(self.gamma, [end_state.gamma, end_state.gamma], 0)
            new_mass_ratios[-1, :] = end_state.mass_ratios
            new_mass_ratios[-2, :] = end_state.mass_ratios
        self.mass_ratios = new_mass_ratios

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
            dt_over_dx = dt / self.dx
            self.density_fluxes, \
            self.momentum_fluxes, \
            self.total_energy_fluxes = FluxCalculator1D.calculate_muscl_fluxes(self.densities,
                                                                               self.pressures,
                                                                               self.velocities,
                                                                               self.gamma,
                                                                               dt_over_dx)
        else:
            raise RuntimeError("Flux calculator does not exist")

        grid_length = len(self.densities)
        if self.flux_calculator != FluxCalculator1D.MUSCL:
            self.densities = self.densities[1:grid_length - 1]
            self.pressures = self.pressures[1:grid_length - 1]
            self.velocities = self.velocities[1:grid_length - 1]
            self.internal_energies = self.internal_energies[1:grid_length - 1]
            self.gamma = self.gamma[1:grid_length - 1]
            self.mass_ratios = self.mass_ratios[1:grid_length - 1]
        else:
            self.densities = self.densities[2:grid_length - 2]
            self.pressures = self.pressures[2:grid_length - 2]
            self.velocities = self.velocities[2:grid_length - 2]
            self.internal_energies = self.internal_energies[2:grid_length - 2]
            self.gamma = self.gamma[2:grid_length - 2]
            self.mass_ratios = self.mass_ratios[2:grid_length - 2]

    def _calculate_time_step(self):
        """
        Calculates the time step from the approximation in Toro 6 using max(|u| + a) in the domain
        """
        max_wave_speed = 0.0
        for i, dens in enumerate(self.densities):
            sound_speed = np.sqrt(self.gamma[i] * self.pressures[i] / self.densities[i])
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
                gamma = self.gamma[i]

                velocity = momentum / rho
                kinetic_energy = 0.5 * rho * velocity ** 2
                internal_energy = total_energy - kinetic_energy
                pressure = internal_energy * (gamma - 1)

                state = ThermodynamicState1D(pressure, rho, velocity, gamma)
            else:
                total_density_flux = (self.density_fluxes[i] - self.density_fluxes[i + 1]) * dt / self.dx
                total_momentum_flux = (self.momentum_fluxes[i] - self.momentum_fluxes[i + 1]) * dt / self.dx
                total_energy_flux = (self.total_energy_fluxes[i] - self.total_energy_fluxes[i + 1]) * dt / self.dx

                state = ThermodynamicState1D(self.pressures[i], self.densities[i], self.velocities[i], self.gamma[i])
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
        times = [t]
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

        self.validate()

    def validate(self):
        """
        Check that variables passed to Controller by simulation are the correct types and sizes
        """
        assert isinstance(self.x, np.ndarray) and self.x.ndim == 1
        assert isinstance(self.y, np.ndarray) and self.x.ndim == 1
        assert isinstance(self.densities, np.ndarray) and self.densities.ndim == 2
        assert isinstance(self.vel_x, np.ndarray) and self.vel_x.ndim == 2
        assert isinstance(self.vel_y, np.ndarray) and self.vel_y.ndim == 2
        assert isinstance(self.pressures, np.ndarray) and self.pressures.ndim == 2
        assert isinstance(self.internal_energies, np.ndarray) and self.internal_energies.ndim == 2
        assert isinstance(self.gamma, np.ndarray) and self.gamma.ndim == 2

        assert self.x.shape[0] == self.densities.shape[0]
        assert self.y.shape[0] == self.densities.shape[1]
        assert self.densities.shape == self.vel_x.shape == self.vel_y.shape
        assert self.vel_y.shape == self.pressures.shape == self.internal_energies.shape

        assert isinstance(self.dx, float)
        assert isinstance(self.dy, float)
        assert isinstance(self.CFL, float)
        assert isinstance(self.final_time, float)

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
        new_gammas = np.zeros((i_length + 2, j_length + 2))
        new_densities[1:-1, 1:-1] = self.densities
        new_pressures[1:-1, 1:-1] = self.pressures
        new_vel_x[1:-1, 1:-1] = self.vel_x
        new_vel_y[1:-1, 1:-1] = self.vel_y
        new_internal_energies[1:-1, 1:-1] = self.internal_energies
        new_gammas[1:-1, 1:-1] = self.gamma

        # Handle x boundaries
        for j in range(j_length):
            start_state = ThermodynamicState2D(self.pressures[0, j], self.densities[0, j],
                                               self.vel_x[0, j], self.vel_y[0, j], self.gamma[0, j])
            start_state = self.boundary_functions[BoundaryConditionND.X_LOW](start_state, self.y[j])
            new_densities[0, j + 1] = start_state.rho
            new_pressures[0, j + 1] = start_state.p
            new_vel_x[0, j + 1] = start_state.u
            new_vel_y[0, j + 1] = start_state.v
            new_internal_energies[0, j + 1] = start_state.e_int
            new_gammas[0, j + 1] = start_state.gamma

            end_state = ThermodynamicState2D(self.pressures[-1, j], self.densities[-1, j], self.vel_x[-1, j],
                                             self.vel_y[-1, j], self.gamma[-1, j])
            end_state = self.boundary_functions[BoundaryConditionND.X_HIGH](end_state, self.y[j])
            new_densities[-1, j + 1] = end_state.rho
            new_pressures[-1, j + 1] = end_state.p
            new_vel_x[-1, j + 1] = end_state.u
            new_vel_y[-1, j + 1] = end_state.v
            new_internal_energies[-1, j + 1] = end_state.e_int
            new_gammas[-1, j + 1] = end_state.gamma

        # Handle y boundaries
        for i in range(i_length + 2):
            if i == 0:
                x = self.x[0]
            elif i == i_length + 1:
                x = self.x[i_length - 1]
            else:
                x = self.x[i - 1]
            start_state = ThermodynamicState2D(new_pressures[i, 1], new_densities[i, 1], new_vel_x[i, 1],
                                               new_vel_y[i, 1], new_gammas[i, 1])
            start_state = self.boundary_functions[BoundaryConditionND.Y_LOW](start_state, x)
            new_densities[i, 0] = start_state.rho
            new_pressures[i, 0] = start_state.p
            new_vel_x[i, 0] = start_state.u
            new_vel_y[i, 0] = start_state.v
            new_internal_energies[i, 0] = start_state.e_int
            new_gammas[i, 0] = start_state.gamma

            end_state = ThermodynamicState2D(new_pressures[i, -2], new_densities[i, -2], new_vel_x[i, -2],
                                             new_vel_y[i, -2], new_gammas[i, -2])
            end_state = self.boundary_functions[BoundaryConditionND.Y_HIGH](end_state, x)
            new_densities[i, -1] = end_state.rho
            new_pressures[i, -1] = end_state.p
            new_vel_x[i, -1] = end_state.u
            new_vel_y[i, -1] = end_state.v
            new_internal_energies[i, -1] = end_state.e_int
            new_gammas[i, -1] = end_state.gamma

        # Assign new arrays
        self.densities = new_densities
        self.pressures = new_pressures
        self.vel_x = new_vel_x
        self.vel_y = new_vel_y
        self.internal_energies = new_internal_energies
        self.gamma = new_gammas

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
        self.gamma = self.gamma[1:-1, 1:-1]

    def _calculate_time_step(self):
        """
        Calculates the time step from the approximation in Toro 6 using max(|u| + a) in the domain
        """
        max_wave_speed = 0.0
        i_length, j_length = np.shape(self.densities)
        for i in range(i_length):
            for j in range(j_length):
                sound_speed = np.sqrt(self.gamma[i, j] * self.pressures[i, j] / self.densities[i, j])
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
                                             self.vel_x[i, j], self.vel_y[i, j], self.gamma[i, j])

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