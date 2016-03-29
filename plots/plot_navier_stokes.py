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

    fig = plt.figure(figsize=(13, 10), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, y)
    surf1 = ax.plot_surface(X, Y, v_solution_linear[0, :, :], rstride=1, cstride=1)
    fig.savefig('2d_convection_initial_time.png')

    fig2 = plt.figure(figsize=(13, 10), dpi=100)
    ax2 = fig2.gca(projection='3d')
    X, Y = np.meshgrid(x, y)
    surf2 = ax2.plot_surface(X, Y, v_solution_linear[-1, :, :], rstride=1, cstride=1)
    fig2.savefig('2d_convection_final_time.png')

    u_solution, v_solution = NavierStokes.convection_2d(x, y, t, u_initial, v_initial, linear=False)

    fig = plt.figure(figsize=(13, 10), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, y)
    surf3 = ax.plot_surface(X, Y, u_solution[-1, :, :], rstride=1, cstride=1, cmap=cm.coolwarm)
    fig.savefig('2d_convection_non_linear_u.png')

    fig = plt.figure(figsize=(13, 10), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, y)
    surf4 = ax.plot_surface(X, Y, v_solution[-1, :, :], rstride=1, cstride=1, cmap=cm.coolwarm)
    fig.savefig('2d_convection_non_linear_v.png')


if __name__ == '__main__':
    # plot_1d_convection()
    plot_2d_convection()