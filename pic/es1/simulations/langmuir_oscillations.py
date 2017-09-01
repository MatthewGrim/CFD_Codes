"""
Author: Rohan Ramasamy
Date: 24/08/2017

This file contains a simulation of simple langmuir oscillations using the 1D electrostatic pic code.
"""

import numpy as np
from matplotlib import pyplot as plt

from pic.es1.controller import ES1Controller
from pic.es1.particle_initialiser.particle_initialisers import ParticleDistributionType


def simulate_lagmuir_oscillations():
    # Time variables
    dt = 0.2
    final_time = 30.0

    # Grid variables
    domain_length = 2 * np.pi
    num_cells = 1000

    # Particle variables
    num_particles = 2000
    particle_charge = -1.0
    particle_mass = 1.0

    sim = ES1Controller(domain_length, num_cells, dt, final_time,
                        num_particles, particle_mass, particle_charge,
                        0.0, ParticleDistributionType.PERTURBATION,
                        160)

    sim.initialise_particles()
    sim.run()

    plt.figure()
    plt.plot(sim.times, sim.kinetic_energy, label="Kinetic Energy")
    plt.plot(sim.times, sim.electrostatic_energy, label="Electrostatic Energy")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    simulate_lagmuir_oscillations()
