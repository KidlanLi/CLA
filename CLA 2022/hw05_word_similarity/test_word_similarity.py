from unittest import TestCase
from hw05_word_similarity.cooccurrence import vocabulary_from_wordlist
from hw05_word_similarity.word_similarity import PpmiWeightedSparseMatrix, DenseSimilarityMatrix
from scipy.sparse import spmatrix
import numpy as np


class TestWordSimilarity(TestCase):
    def setUp(self):
        self.list_of_words = ['tree', 'road', 'land', 'land', 'tree', 'road', 'land', 'road', 'tree', 'sea', 'water',
                              'water', 'ship', 'ship', 'sea']

    # Exercise 1
    def test01_vocabulary_from_wordlist(self):
        """ Tests if top n frequent words are chosen correctly from a word-list."""
        v = vocabulary_from_wordlist(['a', 'rose', 'is', 'a', 'rose', 'colour', 'rose'], 2)
        self.assertEqual(v, {'a', 'rose'})
        v2 = vocabulary_from_wordlist(['a', 'rose', 'is', 'a', 'rose'], 4)
        self.assertEqual(v2, {'a', 'rose', 'is'})
        v3 = vocabulary_from_wordlist(['double', 'double', 'dub', 'dub', 'single'], 1)
        self.assertEqual(len(v3), 1)
        v4 = vocabulary_from_wordlist(self.list_of_words, 3)
        self.assertEqual(v4, {'tree', 'road', 'land'})

    # Exercise 2.1
    def test02_create_sparse_matrix(self):
        """ Tests PpmiWeightedSparseMatrix instantiation"""
        m = PpmiWeightedSparseMatrix(word_list=self.list_of_words, vocab_size=3, window_size=1)
        self.assertEqual(m.word_to_id.keys(), {'tree', 'road', 'land'})
        self.assertEqual(set(m.id_to_word.values()), {'tree', 'road', 'land'})
        self.assertEqual(m.id_to_word.keys(), {0, 1, 2})
        self.assertEqual(set(m.word_to_id.values()), {0, 1, 2})
        self.assertIsInstance(m.word_matrix, spmatrix)
        self.assertEqual(m.word_matrix.shape, (3, 3))
        self.assertAlmostEqual(m.word_matrix.sum(), 1.96, delta=0.01)

    # Exercise 2.2
    def test03_toSvdSimilarityMatrix(self):
        """ Tests DenseSimilarityMatrix instantiation"""
        m_sparse = PpmiWeightedSparseMatrix(word_list=self.list_of_words, vocab_size=6, window_size=1)
        m_svd = m_sparse.toSvdSimilarityMatrix(n_components=2)
        self.assertIsInstance(m_svd, DenseSimilarityMatrix)
        self.assertIsInstance(m_svd.word_matrix, np.ndarray)
        self.assertEqual(m_svd.word_matrix.shape, (6, 2))
        self.assertAlmostEqual(m_svd.word_similarity('road', 'land'), 0.996, delta=0.05)
        self.assertAlmostEqual(m_svd.word_similarity('road', 'water'), 0.057, delta=0.05)
        self.assertAlmostEqual(m_svd.word_similarity('ship', 'water'), 1.0, delta=0.05)

    # Exercise 2.3
    def test04a_most_similar_words(self):
        """ Tests svd ranking"""
        m_sparse = PpmiWeightedSparseMatrix(word_list=self.list_of_words, vocab_size=6, window_size=1)
        m_svd = m_sparse.toSvdSimilarityMatrix(n_components=2)
        self.assertIsNotNone(m_svd)
        self.assertEqual(m_svd.most_similar_words('land', 3), ['land', 'road', 'tree'])

    # Exercise 2.3
    def test04b_most_similar_words(self):
        """ Tests sparse ranking"""
        m_sparse = PpmiWeightedSparseMatrix(word_list=self.list_of_words, vocab_size=6, window_size=1)
        self.assertEqual(m_sparse.most_similar_words('land', 3), ['land', 'tree', 'road'])
