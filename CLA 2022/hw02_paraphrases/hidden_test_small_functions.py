from unittest import TestCase
from hw02_paraphrases import small_functions as sf
import numpy as np


class TestSmallFunctions(TestCase):

    # Exercise 2.1
    def test01a_square_roots_hidden(self):
        np.testing.assert_almost_equal([ 0], sf.square_roots(0, 2, 1))

    # Exercise 2.1
    def test01b_square_roots_hidden(self):
        np.testing.assert_almost_equal([1.4142136, 1.], sf.square_roots(2, 1, 2))

    # Exercise 2.2
    def test02a_odd_ones_squared_hidden(self):
        x = [[0]]
        np.testing.assert_equal(x, sf.odd_ones_squared(1, 1))

    # Exercise 2.2
    def test02b_odd_ones_squared_hidden(self):
        x = [[0, 1, 2], [9, 4, 25]]
        np.testing.assert_equal(x, sf.odd_ones_squared(2, 3))

