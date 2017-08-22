"""
Author: Rohan Ramasamy
Date: 13/08/17

This file contains the class used for the initialisation of particles
at the beginning of the implementation of ES1
"""

from enum import Enum


class ParticleDistributionType(Enum):
    UNIFORM = 1


class ParticleInitialisers(object):
    @staticmethod
    def initialise_particles(particle_positions, particle_velocities, mean_velocity, distribution_type, domain_length):
        """
        Set up the particle distribution f(x, v, 0) at the beginning of the simulation.

        :param particle_positions: position of particles in the domain
        :param particle_velocities: position of velocities in the domain
        :param mean_velocity: average velocity of the particles at the beginning of the simulation.
        :param distribution_type: Type of velocity distribution to be initialised at the beginning of the simulation
        :param domain_length: length of the domain
        :return:
        """
        assert particle_positions.size() == particle_velocities.size()

        num_particles = particle_positions.size
        if distribution_type is ParticleDistributionType.UNIFORM:
            particle_spacing = domain_length / num_particles
            for i, pos in enumerate(particle_positions):
                particle_positions[i] = i * particle_spacing
                particle_velocities[i] = mean_velocity
        else:
            raise ValueError("Specified option is not implemented!")