"""
Author: Rohan
Date: 28/03/17

This file contains vector operations
"""

import numpy as np
import math


def magnitude(vector):
    """
    Function to get the magnitude of a vector
    """
    assert isinstance(vector, np.ndarray) and len(vector.shape) == 1
    vector *= vector
    squared_magnitude = vector.sum()
    return math.sqrt(squared_magnitude)


def dot(vector_1, vector_2):
    """
    Perform the dot product vector_1.vector_2
    :return:
    """
    assert isinstance(vector_1, np.ndarray) and vector_1.shape[0] == 3 and len(vector_1.shape) == 1, \
            "Vector must be a 3D"
    assert isinstance(vector_2, np.ndarray) and vector_2.shape[0] == 3 and len(vector_2.shape) == 1, \
            "Vector must be a 3D"

    return (vector_1 * vector_2).sum()


def vector_projection(vector_1, vector_2):
    """
    Project vector 1 onto the direction of vector 2
    """
    assert isinstance(vector_1, np.ndarray) and vector_1.shape[0] == 3 and len(vector_1.shape) == 1, \
            "Vector must be a 3D"
    assert isinstance(vector_2, np.ndarray) and vector_2.shape[0] == 3 and len(vector_2.shape) == 1, \
            "Vector must be a 3D"

    v_2_norm = vector_2 / magnitude(vector_2)
    return dot(vector_1, v_2_norm) * v_2_norm


def cross(vector_1, vector_2):
    """
    Perform the cross product vector_1 X vector_2
    """
    assert isinstance(vector_1, np.ndarray) and vector_1.shape[0] == 3 and len(vector_1.shape) == 1, \
            "Vector must be a 3D"
    assert isinstance(vector_2, np.ndarray) and vector_2.shape[0] == 3 and len(vector_2.shape) == 1, \
            "Vector must be a 3D"
    res = np.zeros(3)
    res[0] = vector_1[1] * vector_2[2] - vector_1[2] * vector_2[1]
    res[1] = vector_1[2] * vector_2[0] - vector_1[0] * vector_2[2]
    res[2] = vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0]

    return res


def rotate_2d(vector, theta):
    """
    Function to rotate a 2D vector by a particular angle
    :return:
    """
    assert isinstance(vector, np.ndarray) and vector.shape[0] == 2 and len(vector.shape) == 1, "Vector must be a 3D"
    assert isinstance(theta, float)

    c = np.cos(theta)
    s = np.sin(theta)

    R = np.asarray([[c, -s], [s, c]])
    return R.dot(vector)


def rotate_3d(vector, rotation_angles):
    """
    Rotate a 3D vector along 3 angles about the principal axes
    """
    assert isinstance(vector, np.ndarray) and vector.shape[0] == 3 and len(vector.shape) == 1, \
            "Vector must be a 3D"
    assert isinstance(rotation_angles, np.ndarray) and rotation_angles.shape[0] == 3 and len(rotation_angles.shape) == 1, \
            "There should be 3 rotation angles"

    c_x = np.cos(rotation_angles[0])
    s_x = np.sin(rotation_angles[0])
    c_y = np.cos(rotation_angles[1])
    s_y = np.sin(rotation_angles[1])
    c_z = np.cos(rotation_angles[2])
    s_z = np.sin(rotation_angles[2])

    R = np.asarray([[c_y * c_z, c_z * s_x * s_y - c_x * s_z, c_x * c_z * s_y + s_x * s_z],
                   [c_y * s_z, c_x * c_z + s_x * s_y * s_z, -c_z * s_x + c_x * s_y * s_z],
                   [-s_y, c_y * s_x, c_x * c_y]])

    return R.dot(vector)
