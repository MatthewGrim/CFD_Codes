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
            return 2.0 * outer.sound_speed() / (self.gamma - 1) * \
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
            return 1.0 / (outer.rho * outer.sound_speed()) * (p_star / outer.p) ** (-(self.gamma + 1) / (2 * self.gamma))
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
        #
        # numerator = left.a + right.a - 0.5 * (self.gamma - 1) * (right.u - left.u)
        # denominator = left.a / (left.p ** ((self.gamma - 1) / (2 * self.gamma))) + \
        #               right.a / (right.p ** ((self.gamma - 1) / (2 * self.gamma)))
        #
        # return (numerator / denominator) ** ((2 * self.gamma) / (self.gamma - 1))

        gamma = self.gamma
        G1 = (gamma - 1.0) / (2.0 * gamma)
        G3 = (2.0 * gamma) / (gamma - 1.0)
        G4 = 2.0 / (gamma - 1.0)
        G5 = 2.0 / (gamma + 1.0)
        G6 = (gamma - 1.0) / (gamma + 1.0)
        G7 = (gamma - 1.0) / 2.0
        CUP = 0.25 * (left.rho + right.rho) * (left.sound_speed() + right.sound_speed())
        PPV = 0.5 * (left.p + right.p) + 0.5 * (left.u - right.u) * CUP
        PPV = max(1e-6, PPV)
        PMIN = min(left.p, right.p)
        PMAX = max(left.p, right.p)
        QMAX = PMAX / PMIN

        if QMAX <= 2.0 and PMIN <= PPV <= PMAX:
            PM = PPV
        else:
            if (PPV < PMIN):
                PQ = (left.p / right.p) ** G1
                UM = (PQ * left.u / left.sound_speed() + right.u / right.sound_speed() + G4 * (PQ - 1.0)) \
                     / (PQ / left.sound_speed() + 1.0 / right.sound_speed())
                PTL = 1.0 + G7 * (left.u - UM) / left.sound_speed()
                PTR = 1.0 + G7 * (UM - right.u) / right.sound_speed()
                PM = 0.5 * (left.p * PTL ** G3 + right.p * PTR ** G3)
            else:
                GEL = np.sqrt((G5 / left.rho) / (G6 * left.p + PPV))
                GER = np.sqrt((G5 / right.rho) / (G6 * right.p + PPV))
                PM = (GEL * left.p + GER * right.p - (right.u - left.u)) / (GEL + GER)
                PM = max(1e-6, PM)
        return PM

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
        if (left_state.sound_speed() + right_state.sound_speed()) * 2.0 / (left_state.gamma - 1) <= right_state.sound_speed() - left_state.u:
            raise RuntimeError("Vacuum State generated")

        p_star = self.__get_p_star(left_state, right_state)
        
        u_star = self.__get_u_star(p_star, left_state, right_state)
        
        return p_star, u_star

    def sample(self, x_over_t, left_state, right_state, p_star, u_star):
        """
        Function used to sample Riemann problem at a specific wave speed, to get the state
        """

        # Find state along wave line
        if u_star < x_over_t:
            # Consider right wave structures
            rho = right_state.rho
            p = right_state.p
            gamma = right_state.gamma
            if right_state.p >= p_star:
                rho_star = rho * (p_star / p) ** (1 / gamma)
                a_star = np.sqrt(gamma * p_star / rho_star)
                wave_right_high = right_state.u + right_state.sound_speed()
                wave_right_low = u_star + a_star
                if wave_right_high < x_over_t:
                    return right_state.p, right_state.u, right_state.rho
                else:
                    if wave_right_low > x_over_t:
                        return p_star, u_star, rho_star
                    else:
                        multiplier = ((2.0 / (gamma + 1)) - (gamma - 1) * (right_state.u - x_over_t) / (right_state.sound_speed() * (gamma + 1))) ** (2.0 / (gamma - 1.0))
                        rho = right_state.rho * multiplier
                        u = (2.0 / (gamma + 1)) * (-right_state.sound_speed() + (gamma - 1) * right_state.u / 2.0 + x_over_t)
                        p = right_state.p * multiplier ** gamma
                        return p, u, rho
            else:
                rho_star = rho * ((p_star / p + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * (p_star / p) + 1))
                wave_right_shock = right_state.u + right_state.sound_speed() * ((gamma + 1) * p_star / (2 * gamma * right_state.p) + (gamma - 1) / (2 * gamma)) ** 0.5
                if wave_right_shock < x_over_t:
                    return right_state.p, right_state.u, right_state.rho
                else:
                    return p_star, u_star, rho_star
        else:
            # Consider left wave structures
            rho = left_state.rho
            p = left_state.p
            gamma = left_state.gamma
            if left_state.p >= p_star:
                rho_star = rho * (p_star / p) ** (1 / gamma)
                a_star = np.sqrt(gamma * p_star / rho_star)
                wave_left_high = left_state.u - left_state.sound_speed()
                wave_left_low = u_star - a_star
                if wave_left_high > x_over_t:
                    return left_state.p, left_state.u, left_state.rho
                else:
                    if wave_left_low < x_over_t:
                        return p_star, u_star, rho_star
                    else:
                        multiplier = ((2.0 / (gamma + 1)) + (gamma - 1) * (left_state.u - x_over_t) / (left_state.sound_speed() * (gamma + 1))) ** (2.0 / (gamma - 1.0))
                        rho = left_state.rho * multiplier
                        u = (2.0 / (gamma + 1)) * (left_state.sound_speed() + (gamma - 1) * left_state.u / 2.0 + x_over_t)
                        p = left_state.p * multiplier ** gamma
                        return p, u, rho
            else:
                rho_star = rho * ((p_star / p + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * (p_star / p) + 1))
                wave_left_shock = left_state.u - left_state.sound_speed() * ((gamma + 1) * p_star / (2 * gamma * left_state.p) + (gamma - 1) / (2 * gamma)) ** 0.5
                if wave_left_shock > x_over_t:
                    return left_state.p, left_state.u, left_state.rho
                else:
                    return p_star, u_star, rho_star


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