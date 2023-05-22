from unittest import TestCase
from hw03_sklearn_paraphrases import sklearn_experiments as ske

#Solution & hidden
class Test_sklearn_experiments(TestCase):
    def setUp(self):
        self.list_of_dicts_1 = [{'a':14, 'b':11, 'z':1}, {'a':11, 'c':13, 'b':11,'z':1}]
        self.list_of_dicts_1b = [{'a':14, 'b':11, 'x':13}, {'a':11, 'c':13, 'b':11, 'y':12}]
        self.list_of_dicts_2 = [{'c':11, 'x':13, 'y':14, 'a':11}, {'b':12, 'a':13}]

    def test_01_m1_hidden(self):
        """ Tests if list of feature dictionaries is correctly transformed to design matrix. Equality check is
        permutation invariant. """
        x = [[ 14., 11., 0., 13., 0.], [11., 11., 13., 0., 12]]
        y = ske.make_matrix1(self.list_of_dicts_1b)
        self.assertIsNotNone(y)
        self.assertCountEqual(y.toarray().tolist(),x)

    def test_02_m2_hidden(self):
        """ Tests if list of feature dictionaries is correctly transformed to design matrix, only using features from
        another list. Equality check is permutation invariant. """
        x = [[ 11.,  0.,  11. , 0.],[ 13.,  12.,  0. , 0.]]
        y = ske.make_matrix2(self.list_of_dicts_1, self.list_of_dicts_2)
        self.assertIsNotNone(y)
        self.assertCountEqual(y.toarray().tolist(),x)

    def test_03_m1_m2_hidden(self):
        """ Tests if two lists of feature dictionaries are correctly transformed to design matrix. Special case:
        dictionary elements in first list are strict subset of second list."""
        x = ske.make_matrix1([{'a':14.0, 'b':11.0, 'c':0.0, 'd':0.0}, {'a':11.0, 'b':11.0, 'c':13.0, 'd':0.0}])
        y = ske.make_matrix2(self.list_of_dicts_1, self.list_of_dicts_1b)
        self.assertIsNotNone(x)
        self.assertIsNotNone(y)
        print(y.toarray().tolist())
        print(x.toarray().tolist())
        self.assertCountEqual(x.toarray().tolist(),y.toarray().tolist())

    def test_04_m1_m2_hidden(self):
        """ Tests if two lists of feature dictionaries are correctly transformed to design matrix. Special case:
        dictionary elements in second list are strict subset of first list."""
        x = ske.make_matrix1(self.list_of_dicts_1)
        y = ske.make_matrix2(self.list_of_dicts_1, self.list_of_dicts_2)
        self.assertIsNotNone(x)
        self.assertIsNotNone(y)
        self.assertEqual(x.shape, y.shape)
        # checks if x is different from y:
        self.assertRaises(AssertionError, self.assertCountEqual, x.toarray().tolist(),y.toarray().tolist())
