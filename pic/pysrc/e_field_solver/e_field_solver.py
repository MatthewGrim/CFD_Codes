"""
Author: Rohan Ramasamy
Date: 19/07/17

This file contains the implementation of a electric field solver, to resolve the potential field across the domain of a
mesh
"""

import numpy as np
from matplotlib import pyplot as plt


def gauss_seidel_solver(phi, rho, dx, dy, epsilon_0=0.1):
    """
    This function uses the Gauss Seidel method to resolve the electric field. Currently the implementation below assumes
    that the outer edge of cells are boundary conditions, and so the gauss seidel method is not applied to them.

    :param phi: Potential
    :param dx: Mesh size in x direction
    :param dy: Mesh size in y direction
    :return:
    """
    assert isinstance(dx, float) and isinstance(dy, float)
    assert isinstance(rho, np.ndarray) and isinstance(phi, np.ndarray)
    assert rho.shape == phi.shape

    tol = np.max(phi * 0.01)
    iteration_tol = 1
    max_iterations = 1000
    iter_num = 0

    phi_init = rho / epsilon_0 * dx ** 2 * dy ** 2
    while iteration_tol > tol and iter_num < max_iterations:
        print("{}: {}".format(iter_num, iteration_tol))
        iteration_tol = 0.0
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                if i == 0 or j == 0 or i == phi.shape[0] - 1 or j == phi.shape[1] - 1:
                    continue

                phi_i_minus = phi[i - 1, j]
                phi_i_plus = phi[i + 1, j]
                phi_j_minus = phi[i, j - 1]
                phi_j_plus = phi[i, j + 1]

                old_phi = phi[i, j]

                phi[i, j] = phi_init[i, j]
                phi[i, j] += dx ** 2 * (phi_j_plus + phi_j_minus) + dy ** 2 * (phi_i_minus + phi_i_plus)
                phi[i, j] /= 2 * (dx ** 2 + dy ** 2)

                iteration_tol += (old_phi - phi[i, j]) ** 2

        iteration_tol = np.sqrt(iteration_tol)
        iter_num += 1

    return phi


def example_field():
    """
    Test a field with boundary values of phi set
    :return:
    """
    nx = ny = 50
    x = np.linspace(0.0, 2, nx)
    y = np.linspace(0.0, 2, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    phi = np.zeros((nx, ny))
    phi[nx/4:3*nx/4, 0] = 10
    phi[nx/4:3*nx/4, nx - 1] = 5
    phi[0, ny/4:3*ny/4] = 2
    phi[ny - 1, ny/4:3*ny/4] = 20

    rho = np.ones((nx, ny)) * 10
    for i in range(nx):
        for j in range(ny):
            rho[i, j] *= np.exp(-((x[i] - 1) ** 2 + (y[j] - 1) ** 2))

    phi = gauss_seidel_solver(phi, rho, dx, dy)

    plt.figure()
    plt.contourf(x, y, phi.transpose(), 100)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    example_field()