"""
Author: Rohan
Date: 29/06/16

This file contains a class used to model a numerical solution to Riemann problems. The implementation of the Riemann
follows that in Toro - Chapter 4.
"""


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
        return (self.gamma + 1) / (self.gamma - 1) * pressure
    
    def __f(self, p_star, p_outer, rho_outer, a_outer):
        """
        p_star: pressure in the star region
        p_outer: pressure outside star region on either the right or left
        rho_outer: density outside the star region on either the right or left
        a: sound speed outside the star region on either the right ofr left
        
        :return: the solution to the function f in the iterative scheme for calculating p star 
        """
        
        if (p_star <= p_outer):
            return 2.0 * a_outer / (self.gamma - 1) * \
                   ((p_star / p_outer) ** ((self.gamma - 1) / (2 * self.gamma)) - 1)
        else:
            A = self.__get_A(rho_outer)
            B = self.__get_B(p_outer)
            
            return (p_star - p_outer) * (A / (p_star + B)) ** 0.5            

    def __f_total(self, u_left, u_right, p_star, p_left, rho_left, a_left,
                   p_right, rho_right, a_right):

        f_r = self.__f(p_star, p_right, rho_right, a_right)
        f_l = self.__f(p_star, p_left, rho_left, a_left)

        return f_r + f_l + (u_right - u_left)

    def __f_derivative(self, p_star, p_outer, rho_outer, a_outer):
        """
        :param p_star:
        :param p_outer:
        :param rho_outer:
        :param a_outer:

        :return: the derivative of the function f in the iterative scheme for calculating p star
        """

        if (p_star <= p_outer):
            return 1.0 / (rho_outer * a_outer) * (p_star / p_outer) ** (-(self.gamma + 1) / (2 * self.gamma))
        else:
            A = self.__get_A(rho_outer)
            B = self.__get_B(p_outer)

            return (1 - (p_star - p_outer) / (2 * (B + p_star))) * (A / (p_star + B)) ** 0.5

    def __f_total_derivative(self, p_star, p_left, rho_left, a_left,
                             p_right, rho_right, a_right):

        f_deriv_r = self.__f_derivative(p_star, p_right, rho_right, a_right)
        f_deriv_l = self.__f_derivative(p_star, p_left, rho_left, a_left)

        return f_deriv_r + f_deriv_l

    def __estimate_p_star(self, p_left, a_left, p_right, a_right, u_left, u_right):
        """

        :return: an estimate for p_star used in the iterative scheme
        """

        numerator = a_left + a_right - 0.5 * (self.gamma - 1) * (u_right - u_left)
        denominator = (a_left / p_left) ** ((self.gamma - 1) / (2 * self.gamma)) + \
                      (a_right / p_right) ** ((self.gamma - 1) / (2 * self.gamma))

        return (numerator / denominator) ** ((2 * self.gamma) / (self.gamma - 1))

    def __get_p_star(self, p_left, rho_left, a_left,
                   p_right, rho_right, a_right, u_left, u_right):
        """

        :return: the pressure in the star region.
        """
        TOL = 1e-6

        p_sol = self.__estimate_p_star(p_left, a_left, p_right, a_right, u_left, u_right)
        delta = 1.0
        while (delta > TOL):
            f_total = self.__f_total(p_sol, p_left, rho_left, a_left, p_right, rho_right, a_right)
            f_total_derivative = self.__f_total_derivative(p_sol, p_left, rho_left, a_left, p_right, rho_right, a_right)

            p_sol -= f_total / f_total_derivative

        return p_sol
    
    def __get_u_star(self, u_left, u_right, p_star, p_left, rho_left, a_left,
                   p_right, rho_right, a_right):
        """

        :return: the velocity in the star region.
        """
        
        return 0.5 * (u_left + u_right) + 0.5 * (self.f(p_star, p_left, rho_left, a_left) +
                                                 self.f(p_star, p_right, rho_right, a_right))
    
    def get_star_states(self, u_left, u_right, p_left, rho_left, a_left,
                   p_right, rho_right, a_right):
        """
        :param u_left: velocity in the left outer region
        :param u_right: velocity in the right outer region
        :param p_star: pressure in the star region
        :param p_left: pressure in the left star region
        :param rho_left: density in the left outer region
        :param a_left: sound speed in the left region
        :param p_right: pressure in the right outer region
        :param rho_right: density in the right outer region
        :param a_right: sound speed in the right outer region

        :return: the star pressure and velocity states
        """
        assert isinstance(u_left, float)
        assert isinstance(u_right, float)
        assert isinstance(p_left, float)
        assert isinstance(rho_left, float)
        assert isinstance(a_left, float)
        assert isinstance(p_right, float)
        assert isinstance(rho_right, float)
        assert isinstance(a_left, float)

        p_star = self.__get_p_star(p_left, rho_left, a_left,
                                 p_right, rho_right, a_right, u_left, u_right)
        
        u_star = self.__get_u_star(u_left, u_right, p_star, p_left, rho_left, a_left,
                                 p_right, rho_right, a_right)
        
        return p_star, u_star