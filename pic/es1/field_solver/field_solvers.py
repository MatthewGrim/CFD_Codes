"""
Author: Rohan Ramasamy
Date: 13/08/17

This file contains the FieldSolvers class used in ES1
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft


epsilon_0 = 1.0


class FieldSolvers(object):
    @staticmethod
    def get_charge_densities(num_cells, particle_positions, Q, dx, domain_length):
        """
        Map the charges of the particles onto the charge densities on the grid

        :param num_cells:
        :param particle_positions: particle positions
        :param Q: species charge
        :param dx: cell size
        :param domain_length: total domain size
        :return:
        """
        assert np.all(particle_positions <= domain_length) and np.all(particle_positions >= 0.0)
        grid_densities = np.zeros(num_cells)

        for x in particle_positions:
            if x <= 0.5 * dx:
                right_ratio = (x + 0.5 * dx) / dx
                left_ratio = 1.0 - right_ratio
                grid_densities[-1] += Q * left_ratio
                grid_densities[0] += Q * right_ratio
            elif x >= domain_length - 0.5 * dx:
                right_ratio = (x - domain_length + 0.5 * dx) / dx
                left_ratio = 1.0 - right_ratio
                grid_densities[-1] += Q * left_ratio
                grid_densities[0] += Q * right_ratio
            else:
                i = (x - 0.5 * dx) // dx
                right_ratio = (x - (i + 0.5) * dx) / dx
                left_ratio = 1.0 - right_ratio
                grid_densities[i] += left_ratio * Q
                grid_densities[i + 1] += right_ratio * Q

        return grid_densities

    @staticmethod
    def solve_EField_Fourier(grid_densities, dx, domain_length):
        """
        Solve possion's equation using the fourier transform of the grid densities

        :param grid_densities:
        :param dx:
        :param domain_length:
        :return:
        """
        #  Apply fourier transform to density distribution
        density_transform = fft(grid_densities)

        # Multiply and invert distribution to get potential field
        k_1 = np.linspace(2 * np.pi / domain_length, np.pi / dx, grid_densities.shape[0] // 2)
        k_2 = np.linspace(-np.pi / dx, -2 * np.pi / domain_length, grid_densities.shape[0] // 2)
        k = np.concatenate((k_1, k_2))

        K_squared = (2.0 / dx * np.sin(k * dx / 2.0)) ** 2

        potential_field = ifft(density_transform / (epsilon_0 * K_squared))

        # Solve E_Field using central difference
        E_field = np.zeros(grid_densities.shape)
        for i, E in enumerate(E_field):
            if i == 0:
                E_field[i] = -(potential_field[i + 1] - potential_field[-1]) / (2 * dx)
            elif i == E_field.shape[0] - 1:
                E_field[i] = -(potential_field[0] - potential_field[i - 1]) / (2 * dx)
            else:
                E_field[i] = -(potential_field[i + 1] - potential_field[i - 1]) / (2 * dx)

        return potential_field, E_field

    @staticmethod
    def solve_EField(grid_densities, dx):
        """
        Solve poisson's equation to get the potential and E field

        :param grid_densities:
        :param dx
        :return:
        """
        potential_field = np.ones(grid_densities.shape)
        potential_field[:] = grid_densities
        phi_new = np.zeros(grid_densities.shape)
        E_field = np.zeros(grid_densities.shape)

        # Solve potential field using Jacobi method
        tol = np.max(np.abs(potential_field)) * 0.01
        iteration_tol = 1
        max_iterations = 10000
        iteration_number = 0
        iteration_hist = np.zeros(max_iterations + 1)
        while iteration_tol > tol or iteration_number < 5:
            iteration_tol = 0.0
            for i, phi in enumerate(potential_field):
                # Assume a periodic boundary
                if i == 0:
                    phi_minus = potential_field[-1]
                    phi_plus = potential_field[i + 1]
                elif i == potential_field.shape[0] - 1:
                    phi_minus = potential_field[i - 1]
                    phi_plus = potential_field[0]
                else:
                    phi_minus = potential_field[i - 1]
                    phi_plus = potential_field[i + 1]

                phi_new[i] = grid_densities[i] / epsilon_0 * dx ** 2
                phi_new[i] += phi_plus + phi_minus
                phi_new[i] /= 2

                iteration_tol += (phi - phi_new[i]) ** 2

            iteration_tol = np.sqrt(iteration_tol) / potential_field.shape[0]
            iteration_hist[iteration_number] = iteration_tol
            iteration_number += 1

            potential_field[:] = phi_new

            if iteration_number > max_iterations:
                print(iteration_hist)
                plt.figure()
                plt.plot(iteration_hist)
                plt.show()
                raise RuntimeError("Maximum Iterations Exceeded!")

        # Solve E_Field
        for i, E in enumerate(E_field):
            if i == 0:
                E_field[i] = -(potential_field[i + 1] - potential_field[-1]) / (2 * dx)
            elif i == E_field.shape[0] - 1:
                E_field[i] = -(potential_field[0] - potential_field[i - 1]) / (2 * dx)
            else:
                E_field[i] = -(potential_field[i + 1] - potential_field[i - 1]) / (2 * dx)

        return potential_field, E_field


# --- TESTS ---
def get_grid_densities_test():
    """
    Simple sinusoidal particle distribution to see if the expected density field is recovered
    """
    num_cells = 200
    domain_length = 2 * np.pi
    dx = domain_length / num_cells

    x = np.linspace(dx / 2, domain_length - dx / 2, num_cells)
    assert dx == x[1] - x[0]

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

    rho = FieldSolvers.get_charge_densities(num_cells, particle_positions, 1.0, dx, domain_length)

    plt.figure()
    plt.plot(x, n, label="Expected Charge Density")
    plt.plot(x, rho, label="Calculated Charge Density")
    plt.plot(x, rho - n, label="Diff")
    plt.legend()
    plt.show()


def get_fields_test():
    """
    Simple sinusoidal density distribution to see if the expected E and potential fields are recovered
    """
    num_x = 100
    domain_length = 2 * np.pi
    dx = 2 * np.pi / num_x
    x = np.linspace(dx / 2.0, domain_length - dx / 2, num_x)
    assert dx == x[1] - x[0]

    rho_0 = 2.0
    rho = rho_0 * np.sin(x)
    E_expected = -rho_0 / epsilon_0 * np.cos(x)
    phi_expected = rho_0 / epsilon_0 * np.sin(x)

    phi, E = FieldSolvers.solve_EField(rho, dx)

    fig, ax = plt.subplots(3)
    ax[0].plot(x, rho)
    ax[0].set_title("Charge Density")
    ax[1].plot(x, E, label="Calculated")
    ax[1].plot(x, E_expected, label="Expected")
    ax[1].plot(x, E - E_expected, label="Diff")
    ax[1].legend()
    ax[1].set_title("E Field")
    ax[2].plot(x, phi, label="Calculated")
    ax[2].plot(x, phi_expected, label="Expected")
    ax[2].plot(x, phi - phi_expected, label="Diff")
    ax[2].set_title("Potential Field")
    ax[2].legend()
    plt.show()


def get_fields_test_fourier():
    """
    Simple sinusoidal density distribution to see if the expected E and potential fields are recovered
    """
    num_x = 100
    domain_length = 2 * np.pi
    dx = 2 * np.pi / num_x
    x = np.linspace(dx / 2.0, domain_length - dx / 2, num_x)
    assert dx == x[1] - x[0]

    rho_0 = 2.0
    rho = rho_0 * np.sin(x)
    E_expected = -rho_0 / epsilon_0 * np.cos(x)
    phi_expected = rho_0 / epsilon_0 * np.sin(x)

    phi, E = FieldSolvers.solve_EField_Fourier(rho, dx, domain_length)

    fig, ax = plt.subplots(3)
    ax[0].plot(x, rho)
    ax[0].set_title("Charge Density")
    ax[1].plot(x, E, label="Calculated")
    ax[1].plot(x, E_expected, label="Expected")
    ax[1].plot(x, (E - E_expected), label="Diff")
    ax[1].legend()
    ax[1].set_title("E Field")
    ax[2].plot(x, phi, label="Calculated")
    ax[2].plot(x, phi_expected, label="Expected")
    ax[2].plot(x, (phi - phi_expected), label="Diff")
    ax[2].set_title("Potential Field")
    ax[2].legend()
    plt.show()


if __name__ == '__main__':
    # get_grid_densities_test()
    # get_fields_test()
    get_fields_test_fourier()