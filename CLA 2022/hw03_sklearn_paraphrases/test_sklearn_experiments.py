from unittest import TestCase
from hw03_sklearn_paraphrases import sklearn_experiments as ske

class Test_sklearn_experiments(TestCase):
    def setUp(self):
        self.list_of_dicts_1 = [{'a':4, 'b':1}, {'a':1, 'c':3, 'b':1}]
        self.list_of_dicts_1b = [{'a':4, 'b':1, 'x':3}, {'a':1, 'c':3, 'b':1, 'y':2}]
        self.list_of_dicts_2 = [{'c':1, 'x':3, 'y':4, 'a':1}, {'b':2, 'a':3}]

    def test_01_m0(self):
        """ Tests if list of feature dictionaries is correctly transformed to design matrix. Equality check is
        permutation invariant. """
        x = [[ 4.,  1.,  0.],[ 1.,  1.,  3.]]
        y = ske.make_matrix1(self.list_of_dicts_1)
        self.assertIsNotNone(y)
        self.assertCountEqual(y.toarray().tolist(),x)

    def test_01_m1(self):
        """ Tests if list of feature dictionaries is correctly transformed to design matrix. Equality check is
        permutation invariant. """
        x = [[ 4., 1., 0., 3., 0.], [1., 1., 3., 0., 2]]
        y = ske.make_matrix1(self.list_of_dicts_1b)
        self.assertIsNotNone(y)
        self.assertCountEqual(y.toarray().tolist(),x)

    def test_02_m2(self):
        """ Tests if list of feature dictionaries is correctly transformed to design matrix, only using features from
        another list. Equality check is permutation invariant. """
        x = [[ 1.,  0.,  1.],[ 3.,  2.,  0.]]
        y = ske.make_matrix2(self.list_of_dicts_1, self.list_of_dicts_2)
        self.assertIsNotNone(y)
        self.assertCountEqual(y.toarray().tolist(),x)

    def test_03_m1_m2(self):
        """ Tests if two lists of feature dictionaries are correctly transformed to design matrix. Special case:
        dictionary elements in first list are strict subset of second list."""
        x = ske.make_matrix1(self.list_of_dicts_1)
        y = ske.make_matrix2(self.list_of_dicts_1, self.list_of_dicts_1b)
        self.assertIsNotNone(x)
        self.assertIsNotNone(y)
        self.assertCountEqual(x.toarray().tolist(),y.toarray().tolist())

    def test_04_m1_m2(self):
        """ Tests if two lists of feature dictionaries are correctly transformed to design matrix. Special case:
        dictionary elements in second list are strict subset of first list."""
        x = ske.make_matrix1(self.list_of_dicts_1b)
        y = ske.make_matrix2(self.list_of_dicts_1b, self.list_of_dicts_1)
        self.assertIsNotNone(x)
        self.assertIsNotNone(y)
        self.assertEqual(x.shape, x.shape)
        # checks if x is different from y:
        self.assertRaises(AssertionError, self.assertCountEqual, x.toarray().tolist(),y.toarray().tolist())
