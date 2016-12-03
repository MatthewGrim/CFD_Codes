"""
Author: Rohan
Date: 01/12/16

This file contains a class to generate a van der corput random number sequence
"""

import numpy as np
import math


class VanDerCorput(object):
    def __init__(self):
        pass

    @staticmethod
    def get_coefficients(n, k_1):
        """
        Generate the coefficients of number n in the sequence
        :param n: the number of the coefficient in the sequence
        :param k_1: coefficient of the van der corput sequence
        :return: array of coefficients a
        """
        assert isinstance(n, int)
        assert isinstance(k_1, int)

        m = np.floor(math.log(n, k_1))

        a = np.zeros(m + 1)
        i = m
        remainder = n
        while i >= 0:
            a[i] = int(remainder / (k_1 ** i))
            remainder = np.mod(remainder, k_1 ** i)
            i -= 1

        return a

    @staticmethod
    def alter_coefficients(a, k_1, k_2):
        assert isinstance(k_1, int)
        assert isinstance(k_2, int)

        i = 0
        while i < a.shape[0]:
            a[i] = np.mod(k_2 * a[i], k_1)
            i += 1
        return a

    @staticmethod
    def calculate_theta(n, k_1, k_2):
        a = VanDerCorput.get_coefficients(n, k_1)
        A = VanDerCorput.alter_coefficients(a, k_1, k_2)

        theta = 0
        for i, A_i in enumerate(A):
            theta += A_i * k_1 ** (-(i + 1))
        return theta

if __name__ == '__main__':
    for i in range(1, 8):
        theta = VanDerCorput.calculate_theta(i, 2, 1)
        print(theta)