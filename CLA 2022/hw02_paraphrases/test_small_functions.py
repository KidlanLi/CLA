from unittest import TestCase
from hw02_paraphrases import small_functions as sf
import numpy as np


class TestSmallFunctions(TestCase):

    # Exercise 2.1
    def test01a_square_roots(self):
        np.testing.assert_almost_equal([2., 2.54950976, 3.], sf.square_roots(4, 9, 3))

    # Exercise 2.1
    def test01b_square_roots(self):
        np.testing.assert_almost_equal([1.], sf.square_roots(1, 2, 1))

    # Exercise 2.2
    def test02a_odd_ones_squared(self):
        x = [[0, 1, 2, 9, 4], [25, 6, 49, 8, 81], [10, 121, 12, 169, 14]]
        np.testing.assert_equal(x, sf.odd_ones_squared(3, 5))

    # Exercise 2.2
    def test02b_odd_ones_squared(self):
        x = [[0, 1], [2, 9]]
        np.testing.assert_equal(x, sf.odd_ones_squared(2, 2))
