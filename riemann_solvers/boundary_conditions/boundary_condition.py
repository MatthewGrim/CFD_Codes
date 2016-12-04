"""
Author: Rohan
Date: 04/12/16

This file contains the implementation of boundary conditions for a Riemann solver based solution to a CFD problem
"""

from CFD_Projects.riemann_solvers.eos.thermodynamic_state import ThermodynamicState


class BoundaryConditionND(object):
    X_LOW = 0
    X_HIGH = 1
    Y_LOW = 2
    Y_HIGH = 3


class BoundaryCondition1D(BoundaryConditionND):
    @staticmethod
    def transmissive_boundary_condition(state):
        assert isinstance(state, ThermodynamicState)

        return state

    @staticmethod
    def reflecting_boundary_condition(state):
        assert isinstance(state, ThermodynamicState)

        return ThermodynamicState(state.p, state.rho, -state.u, state.gamma)