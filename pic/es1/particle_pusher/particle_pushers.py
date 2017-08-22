"""
Author: Rohan Ramasamy
Date: 13/08/17

This file contains the class used to evolve the particle motion in
the implementation of ES1
"""


class ParticlePushers(object):
    @staticmethod
    def advance_time_step(E_field, particle_positions, particle_velocities, particle_charges, particle_masses, dt):
        """
        This function is used to advance particles over a single time step

        :param E_field: function providing the electric field across the domain
        :param partice_positions: the positions of each particle in the domain
        :param particle_velocities: the velocities of each particle in the domain
        :param particle_charges: the charge of each particle species
        :param particle_masses: the mass of each particle species
        :param dt: the duration of the time step
        :param dx: the distance between adjacent cell centres
        :return:
        """
        # Calculate new velocities
        particle_velocities[:] += particle_charges * E_field(particle_positions[:]) / particle_masses * dt

        # Calculate new positions
        particle_positions[:] += particle_velocities * dt

    @staticmethod
    def initialise_velocities(particle_positions, particle_velocities, E_field, Q, particle_masses, dt):
        """
        Initialise velocities one half step behind positions using initial E Field. This is used to set up the
        leap frog scheme.

        :param particle_positions: the positions of each particle in the domain
        :param particle_velocities: the velocities of each particle in the domain
        :param E_field: function to evaluate the electric field at any point in the domain
        :param Q: the particle charge
        :param particle_masses: the particle mass
        :param dt: the time step
        :return:
        """
        particle_velocities[:] -= Q * E_field(particle_positions[:]) / particle_masses * 0.5 * dt