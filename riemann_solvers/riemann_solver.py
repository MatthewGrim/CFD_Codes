"""
Author: Rohan
Date: 29/06/16

This file contains a class used to model a numerical solution to Riemann problems. The implementation of the Riemann
follows that in Toro - Chapter 4.
"""

import numpy as np

from thermodynamic_state import ThermodynamicState


class RiemannSolver(object):
    def __init__(self, gamma):
        assert isinstance(gamma, float)
        self.gamma = gamma
    
    def __get_A(self, rho):
        """
        rho: density in the system outside of the star region on either the right or left.
        
        :return: the coefficient A for the solution of f 
        """
        return 2.0 / ((self.gamma + 1) * rho)
    
    def __get_B(self, pressure):
        """
        pressure: the pressure in the system outside the star region on either the right or left.
        
        :return: the coefficient B for the solution of f 
        """
        return (self.gamma - 1) / (self.gamma + 1) * pressure
    
    def __f(self, p_star, outer):
        """
        p_star: pressure in the star region
        p_outer: pressure outside star region on either the right or left
        rho_outer: density outside the star region on either the right or left
        a: sound speed outside the star region on either the right ofr left
        
        :return: the solution to the function f in the iterative scheme for calculating p star 
        """
        
        if p_star <= outer.p:
            return 2.0 * outer.a / (self.gamma - 1) * \
                   ((p_star / outer.p) ** ((self.gamma - 1) / (2 * self.gamma)) - 1)
        else:
            A = self.__get_A(outer.rho)
            B = self.__get_B(outer.p)
            
            return (p_star - outer.p) * (A / (p_star + B)) ** 0.5

    def __f_total(self, p_star, left, right):

        f_r = self.__f(p_star, right)
        f_l = self.__f(p_star, left)

        return f_r + f_l + (right.u - left.u)

    def __f_derivative(self, p_star, outer):
        """
        :param p_star:
        :param p_outer:
        :param rho_outer:
        :param a_outer:

        :return: the derivative of the function f in the iterative scheme for calculating p star
        """

        if p_star <= outer.p:
            return 1.0 / (outer.rho * outer.a) * (p_star / outer.p) ** (-(self.gamma + 1) / (2 * self.gamma))
        else:
            A = self.__get_A(outer.rho)
            B = self.__get_B(outer.p)

            return (1 - (p_star - outer.p) / (2 * (B + p_star))) * (A / (p_star + B)) ** 0.5

    def __f_total_derivative(self, p_star, left, right):

        f_deriv_r = self.__f_derivative(p_star, right)
        f_deriv_l = self.__f_derivative(p_star, left)

        return f_deriv_r + f_deriv_l

    def __estimate_p_star(self, left, right):
        """

        :return: an estimate for p_star used in the iterative scheme
        """

        numerator = left.a + right.a - 0.5 * (self.gamma - 1) * (right.u - left.u)
        denominator = left.a / (left.p ** ((self.gamma - 1) / (2 * self.gamma))) + \
                      right.a / (right.p ** ((self.gamma - 1) / (2 * self.gamma)))

        return (numerator / denominator) ** ((2 * self.gamma) / (self.gamma - 1))

    def __get_p_star(self, left_state, right_state):
        """

        :return: the pressure in the star region.
        """
        TOL = 1e-6

        p_sol = self.__estimate_p_star(left_state, right_state)
        delta = 1.0
        i = 0
        while delta > TOL:
            f_total = self.__f_total(p_sol, left_state, right_state)
            f_total_derivative = self.__f_total_derivative(p_sol, left_state, right_state)

            p_prev = p_sol
            p_sol -= f_total / f_total_derivative
            delta = np.abs(p_prev - p_sol)
            i += 1

        return p_sol
    
    def __get_u_star(self, p_star, left_state, right_state):
        """

        :return: the velocity in the star region.
        """

        return 0.5 * (left_state.u + right_state.u) + 0.5 * (self.__f(p_star, right_state) - self.__f(p_star, left_state))
    
    def get_star_states(self, left_state, right_state):
        """
        :param left_state: thermodynamic conditions to the left of IVP
        :param right_state: thermodynamic conditions to the right of IVP

        :return: the star pressure and velocity states
        """
        assert isinstance(left_state, ThermodynamicState)
        assert isinstance(right_state, ThermodynamicState)

        # Check for vacuum generation
        if (left_state.a + right_state.a) * 2.0 / (left_state.gamma - 1) <= right_state.a - left_state.u:
            raise RuntimeError("Vacuum state generated")

        p_star = self.__get_p_star(left_state, right_state)
        
        u_star = self.__get_u_star(p_star, left_state, right_state)
        
        return p_star, u_star


def test_iterative_scheme():
    """
    This function runs through the five shock tube problems outlined in Toro - Chapter 4. The results for contact
    velocity, pressure, and number of iterations should match those on p130-131.
    """
    gamma = 1.4
    p_left = [1.0, 0.4, 1000.0, 0.01, 460.894]
    rho_left = [1.0, 1.0, 1.0, 1.0, 5.99924]
    u_left = [0.0, -2.0, 0.0, 0.0, 19.5975]
    p_right = [0.1, 0.4, 0.01, 100.0, 46.0950]
    rho_right = [0.125, 1.0, 1.0, 1.0, 5.99242]
    u_right = [0.0, 2.0, 0.0, 0.0, -6.19633]

    solver = RiemannSolver(1.4)
    print '*' * 50
    for i in range(0, 5):
        print "Riemann Test: " + str(i + 1)

        left_state = ThermodynamicState(p_left[i], rho_left[i], u_left[i], gamma)
        right_state = ThermodynamicState(p_right[i], rho_right[i], u_right[i], gamma)

        p_star, u_star = solver.get_star_states(left_state, right_state)

        print "Converged Star Pressure: " + str(p_star)
        print "Converged Star Velocity: " + str(u_star)
        print '*' * 50


if __name__ == '__main__':
    test_iterative_scheme()