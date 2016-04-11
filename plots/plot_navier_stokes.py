"""
Author: Rohan
Date: 20/03/16
"""
# Standard imports
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

# Project imports
from navier_stokes_model import NavierStokes
from plot_functions import Plot


def plot_1d_convection():
    """
    Function used to plot an example of 1d linear and non-linear convection
    """
    x_max = 2.0
    nx = 1000
    t_max = 0.625
    nt = 1000
    x = np.linspace(0, x_max, nx)
    t = np.linspace(0, t_max, nt)

    u_initial = np.ones(x.shape)
    u_initial[nx/4:nx/2] = 2

    u_solution_linear = NavierStokes.convection_1d(x, t, u_initial, linear=True)
    u_solution_non_linear = NavierStokes.convection_1d(x, t, u_initial, linear=False)

    # plot final time step
    plt.figure()
    plt.plot(u_solution_linear[:, -1], label="Linear")
    plt.plot(u_solution_non_linear[:, -1], label="Non-linear")
    plt.ylim([0.95, 2.1])
    plt.legend()
    plt.title("Velocity vs. x Plot for final time step of simulation")
    plt.ylabel("velocity (m/s)")
    plt.xlabel("x location (m)")
    plt.show()


def plot_2d_convection():
    """
    Function used to plot 2D convection for linear and non-linear cases in 2D
    """
    x_max = 2.0
    y_max = 2.0
    nx = 81
    ny = 81
    t_max = 0.3125
    nt = 100
    x = np.linspace(0, x_max, nx)
    y = np.linspace(0, y_max, ny)
    t = np.linspace(0, t_max, nt)

    u_initial = np.ones((x.shape[0], y.shape[0]))
    u_initial[nx/4:nx/2+1, ny/4:ny/2+1] = 2
    v_initial = np.ones((x.shape[0], y.shape[0]))
    v_initial[nx/4:nx/2+1, ny/4:ny/2+1] = 2

    u_solution_linear, v_solution_linear = NavierStokes.convection_2d(x, y, t, u_initial, v_initial, linear=True)

    Plot.plot2d(x, y, v_solution_linear[0, :, :], '2d_convection_initial_time.png')
    Plot.plot2d(x, y, v_solution_linear[-1, :, :], '2d_convection_final_time.png')

    u_solution, v_solution = NavierStokes.convection_2d(x, y, t, u_initial, v_initial, linear=False)

    Plot.plot2d(x, y, u_solution[-1, :, :], '2d_convection_non_linear_u.png')
    Plot.plot2d(x, y, v_solution[-1, :, :], '2d_convection_non_linear_v.png')


def plot_2d_laplace():
    """
    Function used to plot a simple solution to the Laplace equation in 2D
    """
    nx = 31
    ny = 31
    c = 1
    dx = 2/(nx - 1)
    dy = 2/(ny - 1)

    u = np.zeros((nx, ny))

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)

    u[:, -1] = y

    u_sol = NavierStokes.laplace_2d(x, y, u)

    Plot.plot2d(x, y, u, "laplace_2d_initial")
    Plot.plot2d(x, y, u_sol, "laplace_2d_final")

def plot_2d_poisson():
    """
    Function used to plot a simple solution to the Poisson equation in 2D
    """
    nx = 50
    ny = 50
    xmin = 0
    xmax = 2
    ymin = 0
    ymax = 1

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    u = np.zeros((nx, ny))
    b = np.zeros((nx, ny))

    b[nx/4, ny/4] = 100
    b[3 * nx/4, 3 * ny/4] = -100

    u_sol = NavierStokes.poisson_2d(x, y, u, b)

    Plot.plot2d(x, y, u, "Poisson_initial")
    Plot.plot2d(x, y, u_sol, "Poisson_final")


if __name__ == '__main__':
    # plot_1d_convection()
    # plot_2d_convection()
    # plot_2d_laplace()
    plot_2d_poisson()
