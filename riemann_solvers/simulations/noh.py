"""
Author: Rohan
Date: 04/12/16

This file contains a class used to simulate a 1D Noh problem as outlined in the Tri-Lab test suite
"""

import numpy as np
from matplotlib import pyplot as plt

from CFD_Projects.riemann_solvers.eos.thermodynamic_state import ThermodynamicState
from CFD_Projects.riemann_solvers.simulations.base_simulation import BaseSimulation1D
from CFD_Projects.riemann_solvers.flux_calculator.flux_calculator import FluxCalculator
from CFD_Projects.riemann_solvers.boundary_conditions.boundary_condition import BoundaryConditionND
from CFD_Projects.riemann_solvers.boundary_conditions.boundary_condition import BoundaryCondition1D
from CFD_Projects.riemann_solvers.controller import Controller1D


class AnalyticNoh(object):
    def __init__(self, state, x_max, num_pts):
        assert isinstance(state, ThermodynamicState)
        assert isinstance(x_max, float)
        assert isinstance(num_pts, int)
        # assert state.e_int == 0
        # assert state.p == 0

        self.initial_state = state
        self.num_pts = num_pts
        self.gamma = state.gamma
        self.x = np.linspace(0, x_max, num_pts)
        self.rho = np.zeros(num_pts)
        self.u = np.zeros(num_pts)
        self.p = np.zeros(num_pts)
        self.e = np.zeros(num_pts)

    def get_solution(self, t):
        assert isinstance(t, float)

        v_shock = 0.5 * np.abs(self.initial_state.u) * (self.gamma - 1)
        r_shock = v_shock * t

        for i, x in enumerate(self.x):
            if x <= r_shock:
                # Sim is assumed to be planar
                self.rho[i] = self.initial_state.rho * ((self.gamma + 1) / (self.gamma - 1))
                self.u[i] = 0
                self.e[i] = 0.5 * self.initial_state.u ** 2
                self.p[i] = (self.gamma - 1) * self.rho[i] * self.e[i]
            else:
                self.rho[i] = self.initial_state.rho
                self.u[i] = self.initial_state.u
                self.p[i] = self.initial_state.p
                self.e[i] = self.initial_state.e_int

        return self.x, self.rho, self.u, self.p, self.e


class Noh1D(BaseSimulation1D):
    def __init__(self, initial_state, final_time, CFL, flux_calculator):
        assert(isinstance(initial_state, ThermodynamicState))
        assert(isinstance(CFL, float))
        assert(isinstance(final_time, float))
        assert(0.0 < CFL < 1.0)

        super(Noh1D, self).__init__()

        self.mesh = np.linspace(0.002, 0.4, 100)
        self.dx = self.mesh[0] * 2
        self.final_time = final_time
        self.CFL = CFL
        self.flux_calculator = flux_calculator

        # Initialise physical states
        self.densities = list()
        self.pressures = list()
        self.velocities = list()
        self.internal_energies = list()
        self.gamma = initial_state.gamma
        for x_loc in self.mesh:
            self.densities.append(initial_state.rho)
            self.pressures.append(initial_state.p)
            self.velocities.append(initial_state.u)
            self.internal_energies.append(initial_state.e_int)
        self.densities = np.asarray(self.densities)
        self.pressures = np.asarray(self.pressures)
        self.velocities = np.asarray(self.velocities)
        self.internal_energies = np.asarray(self.internal_energies)

        self.boundary_functions = {
            BoundaryConditionND.X_LOW: lambda state : BoundaryCondition1D.reflecting_boundary_condition(state),
            BoundaryConditionND.X_HIGH: lambda state : BoundaryCondition1D.transmissive_boundary_condition(state)
        }

        self.is_initialised = True


def test_noh():
    """
    This function runs through the tri lab version of the Noh problem. See the Tri Lab verification test suite.
    """
    # Use small pressure value for numerical stability (and to be physically meaningful)
    initial_state = ThermodynamicState(1e-4, 1.0, -1.0, 5.0 / 3.0)

    # Run Noh sim with Godunov and Random Choice
    noh_god = Noh1D(initial_state, final_time=0.3, CFL=0.45, flux_calculator=FluxCalculator.GODUNOV)
    godunov_sim = Controller1D(noh_god)
    (times_god, x_god, densities_god, pressures_god, velocities_god, internal_energies_god) = godunov_sim.run_sim()

    noh_rc = Noh1D(initial_state, final_time=0.3, CFL=0.45, flux_calculator=FluxCalculator.RANDOM_CHOICE)
    rc_sim = Controller1D(noh_rc)
    (times_rc, x_rc, densities_rc, pressures_rc, velocities_rc, internal_energies_rc) = rc_sim.run_sim()

    # Get analytic solution
    noh_test = AnalyticNoh(initial_state, 0.4, 100)
    x_sol, rho_sol, u_sol, p_sol, e_sol = noh_test.get_solution(0.3)

    title = "Noh Test"
    num_plts_x = 2
    num_plts_y = 2
    plt.figure(figsize=(20, 10))
    plt.suptitle(title)
    plt.subplot(num_plts_x, num_plts_y, 1)
    plt.title("Density")
    plt.plot(x_sol, rho_sol)
    plt.scatter(x_god, densities_god, c='g')
    plt.scatter(x_rc, densities_rc, c='r')
    plt.subplot(num_plts_x, num_plts_y, 2)
    plt.title("Velocity")
    plt.plot(x_sol, u_sol)
    plt.scatter(x_god, velocities_god, c='g')
    plt.scatter(x_rc, velocities_rc, c='r')
    plt.subplot(num_plts_x, num_plts_y, 3)
    plt.title("Pressure")
    plt.plot(x_sol, p_sol)
    plt.scatter(x_god, pressures_god, c='g')
    plt.scatter(x_rc, pressures_rc, c='r')
    plt.subplot(num_plts_x, num_plts_y, 4)
    plt.title("Energy")
    plt.plot(x_sol, e_sol)
    plt.scatter(x_god, internal_energies_god, c='g')
    plt.scatter(x_rc, internal_energies_rc, c='r')
    plt.show()


if __name__ == '__main__':
    test_noh()
