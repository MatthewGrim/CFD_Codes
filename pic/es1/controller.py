"""
This file contains the main controller of ES1, used to control the main time step of simulations
"""


import numpy as np
from scipy.interpolate import interp1d

from pic.es1.field_solver.field_solvers import FieldSolvers
from pic.es1.particle_pusher.particle_pushers import ParticlePushers
from pic.es1.particle_initialiser.particle_initialisers import ParticleInitialisers, ParticleDistributionType


class ES1Controller(object):
    """
    This is the main simulation class for ES1 - a 1D electrostatic pic code
    """
    def __init__(self, x_length, num_x, dt, final_time,
                 num_particles, particle_mass, particle_charge,
                 mean_particle_velocity, particle_distribution_type):
        # Initialise space variables
        assert isinstance(int, num_x)
        assert isinstance(float, x_length)
        self.domain_length = x_length
        self.dx = x_length / num_x
        self.cell_locations = np.linspace(0.0, x_length, num_x)
        self.E_field = np.zeros(num_x)
        self.charge_densities = np.zeros(num_x)
        self.potential_field = np.zeros(num_x)

        # Initialise time variables
        assert isinstance(float, final_time)
        assert isinstance(float, dt)
        self.dt = dt
        self.final_time = final_time

        # Initialise particle variables
        assert isinstance(int, num_particles)
        assert isinstance(float, particle_mass)
        assert isinstance(float, particle_charge)
        assert isinstance(float, mean_particle_velocity)
        assert isinstance(ParticleDistributionType, particle_distribution_type)
        self.num_particles = num_particles
        self.particle_masses = particle_mass
        self.particle_charges = particle_charge
        self.particle_velocities = np.zeros(num_particles)
        self.particle_positions = np.zeros(num_particles)

        self.mean_velocity = mean_particle_velocity
        self.distribution_type = particle_distribution_type

        self.particles_initialised = False

    def initialise_particles(self):
        ParticleInitialisers.initialise_particles(self.mean_velocity, self.distribution_type, self.domain_length)

        self.particles_initialised = True

    def _evolve_timestep(self):
        # Construct interpolation function for E field across the domain
        E_field_function = interp1d(self.cell_locations, self.E_field)

        # Apply acceleration due to Efield and advance positions
        ParticlePushers.advance_time_step(E_field_function,
                                          self.particle_positions,
                                          self.particle_velocities,
                                          self.particle_charges,
                                          self.particle_masses,
                                          self.dt)

        # Calculate Charge Densities
        FieldSolvers.get_charge_densities(self.charge_densities, self.particle_positions, self.particle_charges,
                                          self.dx, self.domain_length)

        # Calculate new EFields
        FieldSolvers.solve_EField()

    def run(self):
        assert self.particles_initialised

        time = 0.0
        time_step = 0

        # Subtract initial velocity half step to initialise leapfrog scheme
        FieldSolvers.get_charge_densities(self.charge_densities, self.particle_positions,
                                          self.Q, self.dx, self.domain_length)
        FieldSolvers.solve_EField()
        E_field_function = interp1d(self.cell_locations, self.E_field)
        ParticlePushers.initialise_velocities(self.particle_positions, self.particle_velocities,
                                              E_field_function, self.particle_charges, self.particle_masses, self.dt)
        while time < self.final_time:
            self._evolve_timestep()

            time += self.dt
            time_step += 1



