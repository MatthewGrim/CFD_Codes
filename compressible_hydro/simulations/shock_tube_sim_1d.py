"""
Author: Rohan
Date: 01/09/16

This file contains a class used to simulate a 1D shock tube problem using a Godunov method Riemann solution. The code in
this file should replicate the results from Toro - Chapter 6
"""

import numpy as np
from matplotlib import pyplot as plt

import cProfile, pstats

from CFD_Projects.compressible_hydro.boundary_conditions.boundary_condition import BoundaryCondition1D
from CFD_Projects.compressible_hydro.boundary_conditions.boundary_condition import BoundaryConditionND
from CFD_Projects.compressible_hydro.eos.thermodynamic_state import ThermodynamicState1D
from CFD_Projects.compressible_hydro.flux_calculator.flux_calculator import FluxCalculator1D
from CFD_Projects.compressible_hydro.simulations.analytic_shock_tube import AnalyticShockTube
from CFD_Projects.compressible_hydro.simulations.base_simulation import BaseSimulation1D
from CFD_Projects.compressible_hydro.controller.controller_1d import Controller1D


class ShockTube1D(BaseSimulation1D):
    def __init__(self, left_state, right_state, membrane_location, final_time, CFL, flux_calculator):
        assert(isinstance(left_state, ThermodynamicState1D))
        assert(isinstance(right_state, ThermodynamicState1D))
        assert(isinstance(membrane_location, float))
        assert(isinstance(CFL, float))
        assert(isinstance(final_time, float))
        assert(0.0 < membrane_location < 1.0)
        assert(0.0 < CFL < 1.0)

        super(ShockTube1D, self).__init__()

        self.x = np.linspace(0.005, 0.995, 100)
        self.dx = self.x[0] * 2
        self.final_time = final_time
        self.CFL = CFL
        self.flux_calculator = flux_calculator

        # Initialise physical states
        self.number_of_fluids = 1
        self.molar_masses = np.asarray([29.0])
        self.specific_heats = np.asarray([2.5])
        self.densities = list()
        self.pressures = list()
        self.vel_x = list()
        self.internal_energies = list()
        self.gamma = list()
        self.mass_ratios = np.zeros((self.x.shape[0], self.number_of_fluids))
        for i, x_loc in enumerate(self.x):
            self.gamma.append(left_state.gamma)
            if x_loc < membrane_location:
                self.densities.append(left_state.rho)
                self.pressures.append(left_state.p)
                self.vel_x.append(left_state.u)
                self.internal_energies.append(left_state.e_int)
                self.mass_ratios[i, 0] = 1.0
            else:
                self.densities.append(right_state.rho)
                self.pressures.append(right_state.p)
                self.vel_x.append(right_state.u)
                self.internal_energies.append(right_state.e_int)
                self.mass_ratios[i, 0] = 1.0
        self.densities = np.asarray(self.densities)
        self.pressures = np.asarray(self.pressures)
        self.vel_x = np.asarray(self.vel_x)
        self.internal_energies = np.asarray(self.internal_energies)
        self.gamma = np.asarray(self.gamma)

        self.boundary_functions = {
            BoundaryConditionND.X_LOW: lambda state : BoundaryCondition1D.transmissive_boundary_condition(state),
            BoundaryConditionND.X_HIGH: lambda state : BoundaryCondition1D.transmissive_boundary_condition(state)
        }

        self.is_initialised = True


def example():
    """
    Runs the problems from Toro Chapter 6 to validate this simulation
    """
    gamma = 1.4
    p_left = [1.0, 0.4, 1000.0, 460.894, 1000.0]
    rho_left = [1.0, 1.0, 1.0, 5.99924, 1.0]
    u_left = [0.75, -2.0, 0.0, 19.5975, -19.5975]
    p_right = [0.1, 0.4, 0.01, 46.0950, 0.01]
    rho_right = [0.125, 1.0, 1.0, 5.99242, 1.0]
    u_right = [0.0, 2.0, 0.0, -6.19633, -19.59745]
    membrane_location = [0.3, 0.5, 0.5, 0.4, 0.8]
    end_times = [0.25, 0.15, 0.012, 0.035, 0.012]

    run_lax_wendroff = False
    run_god = True
    run_rc = False
    run_hllc = False
    run_muscl = False
    for i in range(0, 5):
        # Generate profiler
        profile = cProfile.Profile()
        profile.enable()

        # Get initial conditions for shock tube
        left_state = ThermodynamicState1D(p_left[i], rho_left[i], u_left[i], gamma)
        right_state = ThermodynamicState1D(p_right[i], rho_right[i], u_right[i], gamma)

        # Run simulations
        if run_lax_wendroff:
            shock_tube_lax_wendroff = ShockTube1D(left_state, right_state, membrane_location[i],
                                                  final_time=end_times[i], CFL=0.45,
                                                  flux_calculator=FluxCalculator1D.LAX_WENDROFF)
            godunov_sim = Controller1D(shock_tube_lax_wendroff)
            (times_lax_wendroff, x_lax_wendroff, densities_lax_wendroff, pressures_lax_wendroff, velocities_lax_wendroff, internal_energies_lax_wendroff, kinetic_energies_lax_wendroff, mass_ratios_lax_wendroff) = godunov_sim.run_sim()

        if run_god:
            shock_tube_god = ShockTube1D(left_state, right_state, membrane_location[i],
                                         final_time=end_times[i], CFL=0.45,
                                         flux_calculator=FluxCalculator1D.GODUNOV)
            godunov_sim = Controller1D(shock_tube_god)
            (times_god, x_god, densities_god, pressures_god, velocities_god, internal_energies_god, kinetic_energies_god, mass_ratios_god) = godunov_sim.run_sim()

        if run_rc:
            shock_tube_rc = ShockTube1D(left_state, right_state, membrane_location[i],
                                        final_time=end_times[i], CFL=0.45,
                                        flux_calculator=FluxCalculator1D.RANDOM_CHOICE)
            random_choice_sim = Controller1D(shock_tube_rc)
            (times_rc, x_rc, densities_rc, pressures_rc, velocities_rc, internal_energies_rc, kinetic_energies_rc,  mass_ratios_rc) = random_choice_sim.run_sim()

        if run_hllc:
            shock_tube_hllc = ShockTube1D(left_state, right_state, membrane_location[i],
                                          final_time=end_times[i], CFL=0.45,
                                          flux_calculator=FluxCalculator1D.HLLC)
            hllc_sim = Controller1D(shock_tube_hllc)
            (times_hllc, x_hllc, densities_hllc, pressures_hllc, velocities_hllc, internal_energies_hllc, kinetic_energies_hllc,  mass_ratios_hllc) = hllc_sim.run_sim()

        if run_muscl:
            shock_tube_muscl = ShockTube1D(left_state, right_state, membrane_location[i],
                                          final_time=end_times[i], CFL=0.45,
                                          flux_calculator=FluxCalculator1D.MUSCL)
            muscl_sim = Controller1D(shock_tube_muscl)
            (times_muscl, x_muscl, densities_muscl, pressures_muscl, velocities_muscl, internal_energies_muscl, kinetic_energies_muscl, mass_ratios_muscl) = muscl_sim.run_sim()

        # Get analytic solution
        sod_test = AnalyticShockTube(left_state, right_state, membrane_location[i], 1000)
        x_sol, rho_sol, u_sol, p_sol, e_int_sol, e_kin_sol = sod_test.get_solution(end_times[i], membrane_location[i])

        # Output stats
        profile.disable()
        profile.create_stats()
        with open("sod{}_profile.txt".format(i + 1), 'w') as fp:
            stats = pstats.Stats(profile, stream=fp)
            stats.sort_stats('cumtime')  # cumtime, ncalls, percall
            stats.print_stats()

        # Plot state results
        title = "Sod Test: {}".format(i + 1)
        num_plts_x = 2
        num_plts_y = 3
        plt.figure(figsize=(20, 10))
        plt.suptitle(title)
        plt.subplot(num_plts_x, num_plts_y, 1)
        plt.title("Density")
        plt.plot(x_sol, rho_sol)
        if run_lax_wendroff:
            plt.scatter(x_lax_wendroff, densities_lax_wendroff, c='b')
        if run_god:
            plt.scatter(x_god, densities_god, c='g')
        if run_rc:
            plt.scatter(x_rc, densities_rc, c='r')
        if run_hllc:
            plt.scatter(x_hllc, densities_hllc, c='k')
        if run_muscl:
            plt.scatter(x_muscl, densities_muscl, c='c')
        plt.xlim([0.0, 1.0])
        plt.subplot(num_plts_x, num_plts_y, 2)
        plt.title("Velocity")
        plt.plot(x_sol, u_sol)
        if run_lax_wendroff:
            plt.scatter(x_lax_wendroff, velocities_lax_wendroff, c='b')
        if run_god:
            plt.scatter(x_god, velocities_god, c='g')
        if run_rc:
            plt.scatter(x_rc, velocities_rc, c='r')
        if run_hllc:
            plt.scatter(x_hllc, velocities_hllc, c='k')
        if run_muscl:
            plt.scatter(x_muscl, velocities_muscl, c='c')
        plt.xlim([0.0, 1.0])
        plt.subplot(num_plts_x, num_plts_y, 3)
        plt.title("Pressure")
        plt.plot(x_sol, p_sol)
        if run_lax_wendroff:
            plt.scatter(x_lax_wendroff, pressures_lax_wendroff, c='b')
        if run_god:
            plt.scatter(x_god, pressures_god, c='g')
        if run_rc:
            plt.scatter(x_rc, pressures_rc, c='r')
        if run_hllc:
            plt.scatter(x_hllc, pressures_hllc, c='k')
        if run_muscl:
            plt.scatter(x_muscl, pressures_muscl, c='c')
        plt.xlim([0.0, 1.0])
        plt.subplot(num_plts_x, num_plts_y, 4)
        plt.title("Specific Internal Energy")
        plt.plot(x_sol, e_int_sol)
        if run_lax_wendroff:
            plt.scatter(x_lax_wendroff, internal_energies_lax_wendroff, c='b')
        if run_god:
            plt.scatter(x_god, internal_energies_god, c='g')
        if run_rc:
            plt.scatter(x_rc, internal_energies_rc, c='r')
        if run_hllc:
            plt.scatter(x_hllc, internal_energies_hllc, c='k')
        if run_muscl:
            plt.scatter(x_muscl, internal_energies_muscl, c='c')
        plt.xlim([0.0, 1.0])
        plt.subplot(num_plts_x, num_plts_y, 5)
        plt.title("Specific Kinetic Energy")
        plt.plot(x_sol, e_kin_sol)
        if run_lax_wendroff:
            plt.scatter(x_lax_wendroff, kinetic_energies_lax_wendroff, c='b')
        if run_god:
            plt.scatter(x_god, kinetic_energies_god / densities_god, c='g')
        if run_rc:
            plt.scatter(x_rc, kinetic_energies_rc / densities_rc, c='r')
        if run_hllc:
            plt.scatter(x_hllc, kinetic_energies_hllc / densities_hllc, c='k')
        if run_muscl:
            plt.scatter(x_muscl, kinetic_energies_muscl / densities_muscl, c='c')
        plt.xlim([0.0, 1.0])
        plt.subplot(num_plts_x, num_plts_y, 6)
        plt.title("Mass Ratios")
        if run_lax_wendroff:
            plt.scatter(x_lax_wendroff, mass_ratios_lax_wendroff, c='b')
        if run_god:
            plt.plot(x_god, mass_ratios_god, c='g')
        if run_rc:
            plt.plot(x_rc, mass_ratios_rc, c='r')
        if run_hllc:
            plt.plot(x_hllc, mass_ratios_hllc, c='k')
        if run_muscl:
            plt.plot(x_muscl, mass_ratios_muscl, c='c')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.show()


if __name__ == '__main__':
    example()