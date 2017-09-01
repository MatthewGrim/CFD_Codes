"""
Author: Rohan Ramasamy
Date: 13/08/17

This file contains the class used for the initialisation of particles
at the beginning of the implementation of ES1
"""

from enum import Enum
import numpy as np
from matplotlib import pyplot as plt


class ParticleDistributionType(Enum):
    UNIFORM = 1
    PERTURBATION = 2


class ParticleInitialisers(object):
    @staticmethod
    def initialise_particles(particle_positions, particle_velocities, mean_velocity, distribution_type, domain_length, **kwargs):
        """
        Set up the particle distribution f(x, v, 0) at the beginning of the simulation.

        :param particle_positions: position of particles in the domain
        :param particle_velocities: position of velocities in the domain
        :param mean_velocity: average velocity of the particles at the beginning of the simulation.
        :param distribution_type: Type of velocity distribution to be initialised at the beginning of the simulation
        :param domain_length: length of the domain
        :return:
        """
        assert particle_positions.size == particle_velocities.size

        num_particles = particle_positions.size
        if distribution_type is ParticleDistributionType.UNIFORM:
            particle_spacing = domain_length / num_particles
            for i, pos in enumerate(particle_positions):
                particle_positions[i] = i * particle_spacing
                particle_velocities[i] = mean_velocity
        elif distribution_type is ParticleDistributionType.PERTURBATION:
            perturbation_amplitude = kwargs.get("perturbation_amplitude", 0.01)
            perturbation_velocity = kwargs.get("perturbation_velocity", 0.0)
            particle_spacing = domain_length / num_particles
            # Construct uniform base field
            for i, pos in enumerate(particle_positions):
                particle_positions[i] = i * particle_spacing
                particle_velocities[i] = mean_velocity
            # Add perturbation
            for i, pos in enumerate(particle_positions):
                particle_velocities[i] += perturbation_velocity * np.cos(particle_positions[i])
                particle_positions[i] += perturbation_amplitude * np.sin(particle_positions[i])

                if particle_positions[i] < 0.0:
                    particle_positions += domain_length
                if particle_positions[i] > domain_length:
                    particle_positions -= domain_length
        else:
            raise ValueError("Specified option is not implemented!")


def uniform_distribution_test():
    """
    Tests the implementation of a simple uniform distribution of particles
    """
    num_particles = 100
    particle_positions = np.zeros(num_particles)
    particle_velocities = np.zeros(num_particles)

    ParticleInitialisers.initialise_particles(particle_positions, particle_velocities, 3.0, ParticleDistributionType.UNIFORM, 2 * np.pi)

    fig, ax = plt.subplots(2)
    ax[0].plot(particle_positions[1:] - particle_positions[:-1])
    ax[1].plot(particle_velocities)
    plt.show()


def perturbed_distribution_test():
    """
    Tests the implementation of a simple perturbed distribution of particles
    """
    num_particles = 100
    particle_positions = np.zeros(num_particles)
    particle_velocities = np.zeros(num_particles)

    ParticleInitialisers.initialise_particles(particle_positions, particle_velocities, 0.0,
                                              ParticleDistributionType.PERTURBATION, 2 * np.pi,
                                              perturbation_amplitude=0.5, perturbation_velocity=0.001)

    fig, ax = plt.subplots(2)
    ax[0].plot(particle_positions)
    ax[1].plot(particle_velocities)
    plt.show()


if __name__ == '__main__':
    uniform_distribution_test()
    perturbed_distribution_test()