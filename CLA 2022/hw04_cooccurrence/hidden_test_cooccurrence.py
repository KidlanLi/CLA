from unittest import TestCase
from scipy.sparse import lil_matrix
import numpy as np
from hw04_cooccurrence import cooccurrence


class CoocurrenceTest(TestCase):
    def setUp(self):
        self.text = "a b a b b b b b a c c a a a b a a b b c b b b c b b"

    def test_01_cooccurrences(self):
        n = 2
        expected = {('a', 'b'): 13, ('a', 'a'): 12, ('b', 'a'): 13, ('b', 'b'): 30, ('b', 'c'): 9, ('a', 'c'): 5, \
                    ('c', 'b'): 9, ('c', 'a'): 5, ('c', 'c'): 2}
        self.assertEqual(cooccurrence.cooccurrences(self.text, n),expected)

    def test_02_cooc_dict_to_matrix(self):
        cooc_dict = {('b', 'a'): 13, ('a', 'b'): 13, ('a', 'c'): 5, ('c', 'b'): 9, ('b', 'c'): 9, ('c', 'a'): 5}
        m = lil_matrix([[0.,13.,5.],[13.,0.,9.],[5.,9.,0.]])
        np.testing.assert_array_equal(cooccurrence.cooc_dict_to_matrix(cooc_dict)[0].toarray(),m.toarray())

    def test_03_ppmi_weight(self):
        m = lil_matrix([[0.,13.,5.],[13.,0.,9.],[5.,9.,0.]])
        ppmi_matrix = lil_matrix([[0., 0.572519, 0.068993],[0.572519, 0., 0.456109],[0.068993, 0.456109, 0.]])
        np.testing.assert_array_almost_equal(cooccurrence.ppmi_weight(m).toarray(),ppmi_matrix.toarray())
