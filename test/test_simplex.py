from unittest import TestCase

from simplex.simplex import simplex
import numpy as np


class TestSimplex(TestCase):
    def test_simplex_1(self):
        (z, x) = simplex('min', np.array([[3, 1], [4, 3], [1, 2]]), np.array([[3], [6], [4]]), np.array([[4], [1]]),
                         np.array([[0], [-1], [1]]), 100)

        self.assertTrue(np.allclose(z, 3.4) and np.allclose(x, np.array([[0.4], [1.8], [0], [1], [0], [0]])))

    def test_simplex_2(self):
        (z, x) = simplex('max', np.array([[6, 4], [1, 2], [-1, 1], [0, 1]]), np.array([[24], [6], [1], [2]]),
                         np.array([[5], [4]]), np.array([[1], [1], [1], [1]]), 100)

        self.assertTrue(np.allclose(z, 21) and np.allclose(x, np.array([[3], [1.5], [0], [0], [2.5], [0.5]])))

    def test_simplex_3(self):
        (z, x) = simplex('min', np.array([[1/2, 1/4], [1, 3], [1, 1]]), np.array([[4], [20], [10]]),
                         np.array([[2], [3]]), np.array([[1], [-1], [0]]), 100)

        self.assertTrue(np.allclose(z, 25) and np.allclose(x, np.array([[5], [5], [0.25], [0], [0], [0]])))

    def test_simplex_4(self):
        (z, x) = simplex('max', np.array([[8, 6, 1], [4, 2, 1.5], [2, 1.5, 0.5], [0, 1, 0]]),
                         np.array([[48], [20], [8], [5]]), np.array([[60], [30], [20]]), np.array([[1], [1], [1], [1]]),
                         100)

        self.assertTrue(np.allclose(z, 280) and np.allclose(x, np.array([[2], [0], [8], [24], [0], [0], [5]])))

    # infeasible solution
    #def test_simplex_5(self):
    #    (z, x) = simplex('max', np.array([[1, 2], [3, 2]]), np.array([[2], [12]]), np.array([[2], [3]]),
    #                     np.array([[1], [-1]]), 100)
