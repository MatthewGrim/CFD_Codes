"""
Author: Rohan
Date: 27/03/17

Tests for the ideal mhd fluid solver
"""

import unittest

from CFD_Projects.mhd.pysrc.applets.vector_ops import *


class VectorOpsTest(unittest.TestCase):
    def test_magnitude(self):
        self.assertEqual(0.0, magnitude(np.zeros(3)))
        self.assertEqual(np.sqrt(3.0), magnitude(np.ones(3)))
        self.assertEqual(3.0, magnitude(np.asarray([1.0, 2.0, -2.0])))

    def test_dot(self):
        self.assertEqual(0.0, dot(np.zeros(3), np.ones(3)))
        self.assertEqual(0.0, dot(np.ones(3), np.zeros(3)))
        self.assertEqual(3.0, dot(np.asarray([0.0, 0.0, 3.0]), np.asarray([0.0, 0.0, 1.0])))
        self.assertEqual(0.0, dot(np.asarray([0.0, 0.1, 0.0]), np.asarray([0.0, 0.0, 1.0])))

    def test_vector_projection(self):
        res_1 = vector_projection(np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0]))
        for res in res_1:
            self.assertEqual(0.0, res)

    def test_cross(self):
        res_1 = cross(np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0]))
        self.assertEqual(1.0, res_1[0])
        self.assertEqual(0.0, res_1[1])
        self.assertEqual(0.0, res_1[2])

        res_2 = cross(np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 1.0, 0.0]))
        self.assertEqual(0.0, res_2[0])
        self.assertEqual(0.0, res_2[1])
        self.assertEqual(0.0, res_2[2])

    def test_rotate_2d(self):
        vector = np.asarray([1.0, 0.0])

        res_1 = rotate_2d(vector, np.pi / 2.0)
        self.assertAlmostEqual(0.0, res_1[0])
        self.assertAlmostEqual(1.0, res_1[1])

        res_2 = rotate_2d(vector, np.pi)
        self.assertAlmostEqual(-1.0, res_2[0])
        self.assertAlmostEqual(0.0, res_2[1])

        res_3 = rotate_2d(vector, -np.pi / 2.0)
        self.assertAlmostEqual(0.0, res_3[0])
        self.assertAlmostEqual(-1.0, res_3[1])

    def test_rotate_3d(self):
        vector = np.asarray([1.0, 0.0, 0.0])
        res_1 = rotate_3d(vector, np.asarray([np.pi / 2.0, 0.0, 0.0]))
        self.assertAlmostEqual(1.0, res_1[0])
        self.assertAlmostEqual(0.0, res_1[1])
        self.assertAlmostEqual(0.0, res_1[2])

        res_2 = rotate_3d(vector, np.asarray([0.0, np.pi / 2.0, 0.0]))
        self.assertAlmostEqual(0.0, res_2[0])
        self.assertAlmostEqual(0.0, res_2[1])
        self.assertAlmostEqual(-1.0, res_2[2])

        res_3 = rotate_3d(vector, np.asarray([0.0, 0.0, np.pi / 2.0]))
        self.assertAlmostEqual(0.0, res_3[0])
        self.assertAlmostEqual(1.0, res_3[1])
        self.assertAlmostEqual(0.0, res_3[2])

        vector = np.asarray([0.0, 1.0, 0.0])
        res_1 = rotate_3d(vector, np.asarray([np.pi / 2.0, 0.0, 0.0]))
        self.assertAlmostEqual(0.0, res_1[0])
        self.assertAlmostEqual(0.0, res_1[1])
        self.assertAlmostEqual(1.0, res_1[2])

        res_2 = rotate_3d(vector, np.asarray([0.0, np.pi / 2.0, 0.0]))
        self.assertAlmostEqual(0.0, res_2[0])
        self.assertAlmostEqual(1.0, res_2[1])
        self.assertAlmostEqual(0.0, res_2[2])

        res_3 = rotate_3d(vector, np.asarray([0.0, 0.0, np.pi / 2.0]))
        self.assertAlmostEqual(-1.0, res_3[0])
        self.assertAlmostEqual(0.0, res_3[1])
        self.assertAlmostEqual(0.0, res_3[2])


if __name__ == '__main__':
    unittest.main()