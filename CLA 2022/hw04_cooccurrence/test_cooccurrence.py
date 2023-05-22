from unittest import TestCase
from scipy.sparse import lil_matrix

import numpy as np

from hw04_cooccurrence.cooccurrence import cooccurrences, cooc_dict_to_matrix, ppmi_weight


class CooccurrenceTest(TestCase):

    def setUp(self):
        pass

    def test_01_cooccurrences_01(self):
        """Verify that the correct cooccurrences are calculated"""
        cooc_dict_is = cooccurrences("a rose is a rose", 2)
        cooc_dict_should_be = {('rose', 'is'): 2, ('rose', 'a'): 3, ('a', 'rose'): 3, ('a', 'is'): 2, ('is', 'rose'): 2, ('is', 'a'): 2}
        self.assertEqual(cooc_dict_is, cooc_dict_should_be)
        cooc_dict_is = cooccurrences("A rose is a rose.", 1)
        cooc_dict_should_be = {('rose', 'is'): 1, ('rose', 'a'): 2, ('a', 'rose'): 2, ('is', 'rose'): 1, ('a', 'is'): 1, ('.', 'rose'): 1, ('rose', '.'): 1, ('is', 'a'): 1}
        self.assertEqual(cooc_dict_is, cooc_dict_should_be)

    def test_01_cooccurrences_02(self):
        """Verify that the correct cooccurrences are calculated"""
        cooc_dict_is = cooccurrences("John loves Mary", 5)
        cooc_dict_should_be = {('mary', 'loves'): 1, ('john', 'mary'): 1, ('mary', 'john'): 1, ('loves', 'john'): 1, ('loves', 'mary'): 1, ('john', 'loves'): 1}
        self.assertEqual(cooc_dict_is, cooc_dict_should_be)
        cooc_dict_is = cooccurrences("a man is a man", 4)
        cooc_dict_should_be = {('a', 'man'): 4, ('a', 'is'): 2, ('a', 'a'): 2, ('man', 'a'): 4, ('man', 'is'): 2, ('man', 'man'): 2, ('is', 'a'): 2, ('is', 'man'): 2}
        self.assertEqual(cooc_dict_is, cooc_dict_should_be)

    def test_02_cooc_dict_to_matrix_01(self):
        """Verify that matrix is created correctly"""
        d = {('rose', 'is'): 2, ('rose', 'a'): 3, ('a', 'rose'): 3, ('a', 'is'): 2, ('is', 'rose'): 2, ('is', 'a'): 2}
        m_is, w2id_is = cooc_dict_to_matrix(d)
        nnz_is = m_is.nnz
        nnz_should_be = 6
        m_is = m_is.toarray()
        w2id_should_be = {'is': 1, 'a': 0, 'rose': 2}
        m_should_be = np.array([[ 0.,  2.,  3.], [ 2.,  0.,  2.], [ 3.,  2.,  0.]])
        self.assertEqual(nnz_is, nnz_should_be)
        self.assertEqual(w2id_is, w2id_should_be)
        np.testing.assert_array_equal(m_is, m_should_be)

    def test_02_cooc_dict_to_matrix_02(self):
        """Verify that matrix is created correctly"""
        d = {('mary', 'loves'): 1, ('john', 'mary'): 1, ('mary', 'john'): 1, ('loves', 'john'): 1}
        m_is, w2id_is = cooc_dict_to_matrix(d)
        nnz_is = m_is.nnz
        nnz_should_be = 4
        m_is = m_is.toarray()
        w2id_should_be = {'john': 0, 'loves': 1, 'mary': 2}
        m_should_be = np.array([[ 0.,  0.,  1.], [ 1.,  0.,  0.], [ 1.,  1.,  0.]])

        self.assertEqual(nnz_is, nnz_should_be)
        self.assertEqual(w2id_is, w2id_should_be)
        np.testing.assert_array_equal(m_is, m_should_be)

    def test_03_ppmi_weight_01(self):
        """Verify that PPMI is calculated correctly"""
        m = lil_matrix([[1,2],[3,4]])
        ppmi_weight_is = ppmi_weight(m).toarray()
        ppmi_weight_should_be = np.array([[0., 0.10536052], [ 0.06899287, 0.]])
        np.testing.assert_allclose(ppmi_weight_is, ppmi_weight_should_be, rtol=1e-6)

    def test_03_ppmi_weight_02(self):
        """Verify that PPMI is calculated correctly"""
        m = lil_matrix([[5,6],[7,8]])
        ppmi_weight_is = ppmi_weight(m).toarray()
        ppmi_weight_should_be = np.array([[ 0., 0.0129034], [ 0.01104984, 0.]])
        np.testing.assert_allclose(ppmi_weight_is, ppmi_weight_should_be, rtol=1e-6)

    def test_03_ppmi_weight_03(self):
        """Verify that PPMI is calculated correctly"""
        m = lil_matrix([[1, 8, 4, 4],[2, 0, 2, 1]])
        ppmi_weight_is = ppmi_weight(m).toarray()
        ppmi_weight_should_be = np.array([[ 0., 0.25782911, 0., 0.03468556], [1.07613943, 0., 0.38299225, 0.]])
        np.testing.assert_allclose(ppmi_weight_is, ppmi_weight_should_be, rtol=1e-6)

    def test_03_ppmi_weight_04(self):
        """Verify that PPMI is calculated correctly"""
        m = lil_matrix([[0,1],[1,0]])
        ppmi_weight_is = ppmi_weight(m).toarray()
        ppmi_weight_should_be = np.array([[ 0., 0.69314718], [ 0.69314718, 0.]])
        np.testing.assert_allclose(ppmi_weight_is, ppmi_weight_should_be, rtol=1e-6)
