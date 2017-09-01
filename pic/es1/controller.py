"""
This file contains the main controller of ES1, used to control the main time step of simulations
"""


import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.integrate.quadrature import trapz

from pic.es1.field_solver.field_solvers import FieldSolvers, epsilon_0
from pic.es1.particle_pusher.particle_pushers import ParticlePushers
from pic.es1.particle_initialiser.particle_initialisers import ParticleInitialisers, ParticleDistributionType


class ES1Controller(object):
    """
    This is the main simulation class for ES1 - a 1D electrostatic pic code
    """
    def __init__(self, x_length, num_x, dt, final_time,
                 num_particles, particle_mass, particle_charge,
                 mean_particle_velocity, particle_distribution_type,
                 plot_frequency):
        # Initialise space variables
        assert isinstance(num_x, int)
        assert isinstance(x_length, float)
        self.num_cells = num_x
        self.domain_length = x_length
        self.dx = x_length / num_x
        self.cell_locations = np.linspace(0.0, x_length, num_x)
        self.E_field = np.zeros(num_x)
        self.charge_densities = np.ones(num_x)
        self.potential_field = np.ones(num_x)

        # Initialise time variables
        assert isinstance(final_time, float)
        assert isinstance(dt, float)
        self.dt = dt
        self.final_time = final_time

        # Initialise particle variables
        assert isinstance(num_particles, int)
        assert isinstance(particle_mass, float)
        assert isinstance(particle_charge, float)
        assert isinstance(mean_particle_velocity, float)
        self.num_particles = num_particles
        self.particle_masses = particle_mass
        self.particle_charges = particle_charge
        self.particle_velocities = np.zeros(num_particles)
        self.particle_positions = np.zeros(num_particles)

        self.mean_velocity = mean_particle_velocity
        self.distribution_type = particle_distribution_type

        self.particles_initialised = False

        # Initialise history arrays for storing output data and plotting parameters
        self.times = list()
        self.electrostatic_energy = list()
        self.kinetic_energy = list()

        self.plot_frequency = plot_frequency

    def initialise_particles(self, **kwargs):
        ParticleInitialisers.initialise_particles(self.particle_positions, self.particle_velocities,
                                                  self.mean_velocity, self.distribution_type, self.domain_length,
                                                  **kwargs)

        self.particles_initialised = True

    def _get_diagnostics(self, time):
        """
        Saves salient data such as potential and kinetic energy of system to arrays.

        :param time: Current time
        :return:
        """
        if len(self.times) > 0:
            assert time > self.times[-1]
        else:
            assert time == 0

        self.times.append(time)
        self.electrostatic_energy.append(np.sum(0.5 * epsilon_0 * np.abs(self.E_field) ** 2 * self.dx))
        self.kinetic_energy.append(np.sum(0.5 * self.particle_masses * self.particle_velocities ** 2))

    def _evolve_timestep(self):
        # Construct interpolation function for E field across the domain
        E_field_function = interp1d(self.cell_locations,
                                    self.E_field)

        # Apply acceleration due to Efield and advance positions
        ParticlePushers.advance_time_step(E_field_function,
                                          self.particle_positions,
                                          self.particle_velocities,
                                          self.particle_charges,
                                          self.particle_masses,
                                          self.dt,
                                          self.dx,
                                          self.domain_length)
        assert self.particle_positions.shape[0] == self.num_particles

        # Calculate Charge Densities
        self.charge_densities = FieldSolvers.get_charge_densities(self.num_cells,
                                          self.particle_positions,
                                          self.particle_charges,
                                          self.dx,
                                          self.domain_length)

        # Calculate new EFields
        self.potential_field, self.E_field = FieldSolvers.solve_EField_Fourier(self.charge_densities, self.dx, self.domain_length)

    def run(self):
        assert self.particles_initialised

        time = 0.0
        time_step = 0

        # Subtract initial velocity half step to initialise leapfrog scheme
        self.charge_densities = FieldSolvers.get_charge_densities(self.num_cells, self.particle_positions,
                                                                self.particle_charges, self.dx, self.domain_length)

        self.potential_field, self.E_field = FieldSolvers.solve_EField_Fourier(self.charge_densities, self.dx, self.domain_length)
        E_field_function = interp1d(self.cell_locations, self.E_field)
        ParticlePushers.initialise_velocities(self.particle_positions, self.particle_velocities,
                                              E_field_function, self.particle_charges, self.particle_masses, self.dt)

        self._get_diagnostics(time)
        while time < self.final_time:
            print(time_step)
            if time_step % self.plot_frequency == 0:
                fig, ax = plt.subplots(2, 2)
                ax[0, 0].plot(self.particle_positions)
                ax[1, 0].plot(self.charge_densities)
                ax[0, 1].plot(self.potential_field)
                ax[1, 1].plot(self.E_field)
                ax[0, 0].set_title("x")
                ax[1, 0].set_title("rho")
                ax[0, 1].set_title("phi")
                ax[1, 1].set_title("E")
                plt.show()

            self._evolve_timestep()

            time += self.dt
            time_step += 1

            self._get_diagnostics(time)



