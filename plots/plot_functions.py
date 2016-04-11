"""
Author: Rohan Ramasamy
Date: 09/04/16
This file contains a class to plot data
"""
# Standard Imports
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np


class Plot(object):
    def __init__(self):
        pass

    @staticmethod
    def plot2d(x, y, u, name):
        """
        Function to plot 2D data on a contour plot

        :param x: x locations of plotted points
        :param y: y locations of plotted points
        :param u: State variable to be plotted as contour
        :param name: A string containing the name of the file
        """
        fig = plt.figure(figsize=(13, 10), dpi=100)
        ax = fig.gca(projection = '3d')
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        fig.savefig(name)
