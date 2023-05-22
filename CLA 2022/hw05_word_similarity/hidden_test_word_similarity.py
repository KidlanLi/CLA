from unittest import TestCase
from hw05_word_similarity.cooccurrence import vocabulary_from_wordlist
from hw05_word_similarity.word_similarity import PpmiWeightedSparseMatrix, DenseSimilarityMatrix
from scipy.sparse import spmatrix
import numpy as np


class HiddenTestWordSimilarity(TestCase):
    def setUp(self):
        test_data = 'u x v a u y v b u x v c u y v d u x e'
        self.list_of_words = test_data.split()

    def test01_vocabulary_from_wordlist_hidden(self):
        """ Tests if top n frequent words are chosen correctly from a word-list."""
        v = vocabulary_from_wordlist([], 5)
        self.assertEqual(v, set())
        v2 = vocabulary_from_wordlist(['a', 'rose', 'is', 'a', 'rose'], 8)
        self.assertEqual(v2, {'a', 'rose', 'is'})
        v3 = vocabulary_from_wordlist(self.list_of_words, 9)
        self.assertEqual(v3, {'x', 'y', 'u', 'v', 'a', 'b', 'c', 'd', 'e'})

    def test02_create_sparse_matrix(self):
        """ Tests PpmiWeightedSparseMatrix instantiation"""
        m = PpmiWeightedSparseMatrix(word_list=self.list_of_words, vocab_size=4, window_size=1)
        self.assertEqual(m.word_to_id['u'], 0)
        self.assertEqual(m.word_to_id['y'], 3)
        self.assertEqual(m.id_to_word[0], 'u')
        self.assertEqual(m.id_to_word[3], 'y')
        self.assertIsInstance(m.word_matrix, spmatrix)
        self.assertEqual(m.word_matrix.shape, (4, 4))
        self.assertAlmostEqual(m.word_matrix.sum(), 5.51, delta=0.01)

    def test03_toSvdSimilarityMatrix(self):
        """ Tests DenseSimilarityMatrix instantiation"""
        m_sparse = PpmiWeightedSparseMatrix(word_list=self.list_of_words, vocab_size=4, window_size=1)
        m_svd = m_sparse.toSvdSimilarityMatrix(n_components=2)
        self.assertIsInstance(m_svd, DenseSimilarityMatrix)
        self.assertIsInstance(m_svd.word_matrix, np.ndarray)
        self.assertEqual(m_svd.word_matrix.shape, (4, 2))
        self.assertAlmostEqual(m_svd.word_similarity('x', 'y'), 1.0)

    def test04_most_similar_words(self):
        """ Tests svd ranking"""
        m_sparse = PpmiWeightedSparseMatrix(word_list=self.list_of_words, vocab_size=4, window_size=1)
        m_svd = m_sparse.toSvdSimilarityMatrix(n_components=2)
        self.assertEqual(m_svd.most_similar_words('x', 2), ['x', 'y'])

    def test05_similarities_of_word(self):
        """ Tests sparse ranking"""
        m_sparse = PpmiWeightedSparseMatrix(word_list=self.list_of_words, vocab_size=4, window_size=1)
        sims = m_sparse.similarities_of_word('x')
        self.assertEqual(sims.tolist(), [0.0, 0.0, 1.0, 0.9577647468293286])
