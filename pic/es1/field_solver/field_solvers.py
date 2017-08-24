"""
Author: Rohan Ramasamy
Date: 13/08/17

This file contains the FieldSolvers class used in ES1
"""

import numpy as np
from matplotlib import pyplot as plt


class FieldSolvers(object):
    @staticmethod
    def get_charge_densities(grid_densities, particle_positions, Q, dx, domain_length):
        """
        Map the charges of the particles onto the charge densities on the grid

        :param grid_densities: charge densities on grid cells
        :param particle_positions: particle positions
        :param Q: species charge
        :param dx: cell size
        :param domain_length: total domain size
        :return:
        """
        assert np.all(particle_positions < domain_length) and np.all(particle_positions > 0.0)

        for x in particle_positions:
            if x < 0.5 * dx:
                grid_densities[0] += Q
            elif x > domain_length - 0.5 * dx:
                grid_densities[-1] += Q
            else:
                i = (x - 0.5 * dx) // dx
                left_ratio = (x - (i + 0.5) * dx) / dx
                right_ratio = 1.0 - left_ratio
                grid_densities[i] += left_ratio * Q
                grid_densities[i + 1] += right_ratio * Q


    @staticmethod
    def solve_EField(grid_densities, potential_field, E_field, dx):
        """
        Solve poisson's equation to get the potential and E field

        :param grid_densities:
        :param potential_field:
        :param E_field:
        :param dx
        :return:
        """
        assert grid_densities.shape == potential_field.shape == E_field.shape
        epsilon_0 = 8.85e-12

        # Solve potential field using Gauss Seidel
        tol = np.max(potential_field) * 0.01
        iteration_tol = 1
        max_iterations = 10000
        iteration_number = 0
        while iteration_tol > tol:
            iteration_tol = 0.0
            for i, potential in enumerate(potential_field):
                # Assume a periodic boundary
                if i == 0:
                    phi_minus = potential_field[-1]
                    phi_plus = potential_field[i + 1]
                    phi = potential_field[i]
                elif i == potential_field.shape[0] - 1:
                    phi_minus = potential_field[i - 1]
                    phi_plus = potential_field[0]
                    phi = potential_field[i]
                else:
                    phi_minus = potential_field[i - 1]
                    phi_plus = potential_field[i + 1]
                    phi = potential_field[i]

                potential_field[i] = grid_densities[i] / epsilon_0 * dx ** 2
                potential_field[i] += phi_plus + phi_minus
                potential_field[i] /= 2

                iteration_tol += (phi - potential_field[i]) ** 2

            iteration_tol = np.sqrt(iteration_tol) / potential_field.shape[0]
            iteration_number += 1

            if iteration_number > max_iterations:
                raise RuntimeError("Maximum Iterations Exceeded!")

        # Solve E_Field
        for i, E in enumerate(E_field):
            if i == 0:
                E_field[i] = -(potential_field[i + 1] - potential_field[i]) / dx
            elif i == E_field.shape[0] - 1:
                E_field[i] = -(potential_field[i] - potential_field[i - 1]) / dx
            else:
                E_field[i] = -(potential_field[i + 1] - potential_field[i - 1]) / (2 * dx)


# --- TESTS ---
def get_grid_densities_test():
    """
    Simple sinusoidal particle distribution to see if the expected density field is recovered
    """
    num_cells = 200
    domain_length = 2 * np.pi
    dx = domain_length / num_cells

    rho = np.zeros(num_cells)
    x = np.linspace(dx, domain_length - dx, num_cells)

    n_0 = 100
    n = n_0 * np.abs(np.sin(x))
    particle_positions = list()
    for i, x_loc in enumerate(x):
        number_of_particles_in_cell = int(n[i])
        particle_dx = dx / number_of_particles_in_cell
        for j in range(number_of_particles_in_cell):
            new_loc = x_loc - dx / 2 + j * particle_dx

            particle_positions.append(new_loc)

    particle_positions = np.asarray(particle_positions)

    FieldSolvers.get_charge_densities(rho, particle_positions, 1.0, dx, domain_length)

    plt.figure()
    plt.plot(x, n, label="Expected Charge Density")
    plt.plot(x, rho, label="Calculated Charge Density")
    plt.legend()
    plt.show()


def get_fields_test():
    """
    Simple sinusoidal density distribution to see if the expected E and potential fields are recovered
    """
    num_x = 100
    x = np.linspace(0.0, 2 * np.pi, num_x)
    dx = x[1] - x[0]

    rho_0 = 2.0
    rho = rho_0 * np.sin(x)
    eps_0 = 8.85e-12
    E_expected = -rho_0 / eps_0 * np.cos(x)
    phi_expected = rho_0 / eps_0 * np.sin(x)

    phi = np.ones(x.shape)
    E = np.ones(x.shape)

    FieldSolvers.solve_EField(rho, phi, E, dx)

    fig, ax = plt.subplots(3)
    ax[0].plot(x, rho)
    ax[0].set_title("Charge Density")
    ax[1].plot(x, E, label="Calculated")
    ax[1].plot(x, E_expected, label="Expected")
    ax[1].legend()
    ax[1].set_title("E Field")
    ax[2].plot(x, phi, label="Calculated")
    ax[2].plot(x, phi_expected, label="Expected")
    ax[2].set_title("Potential Field")
    ax[2].legend()
    plt.show()

if __name__ == '__main__':
    get_grid_densities_test()
    get_fields_test()
