from unittest import TestCase
from hw06_skipgram import utils, skipgram
from collections import Counter
from scipy import stats
import types
import numpy as np


class Test_word_similarity(TestCase):
    # def setUp(self):

    def test_01_vocab_map(self):  # 2p
        """ Tests if vocabulary is mapped to matrix row indices, with most frequent words having the smallest ids."""
        v = utils.vocabulary_to_id_for_wordlist(['a', 'rose', 'rose', 'is', 'a', 'rose'], 2)
        self.assertIsNotNone(v)
        self.assertEqual(v, {'rose': 0, 'a': 1})

    def test_02_sigmoid(self):  # 2p
        """ Tests if logistic sigmoid is calculated correctly."""
        self.assertAlmostEqual(utils.sigmoid(0), 0.5, delta=0.001)
        self.assertAlmostEqual(utils.sigmoid(1), 0.731, delta=0.001)
        self.assertAlmostEqual(utils.sigmoid(-1), 0.269, delta=0.001)
        self.assertAlmostEqual(utils.sigmoid(10), 1.0, delta=0.001)
        self.assertAlmostEqual(utils.sigmoid(-10), 0.0, delta=0.001)

    def test_03_negative_samples_0(self):  # 1p
        """ Tests whether positive tuples are created correctly."""
        test_tokens = ["0", "1", "2", "3", "4"]
        vocab_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
        pos_neg_cooccurrences = utils.positive_and_negative_cooccurrences(test_tokens, max_distance=1,
                                                                          neg_samples_factor=100,
                                                                          vocab_to_id=vocab_dict)
        self.assertIsNotNone(pos_neg_cooccurrences)
        with_negatives = sorted(pos_neg_cooccurrences)


        # Count number of positive target words for id "2"

        sum_positives_twos = sum(1 for t in with_negatives if t[2] and
                                 t[0] == 2)
        # Count number of negative target words for id "2"
        sum_negative_twos = sum(1 for t in with_negatives if not t[2] and
                                t[0] == 2)

        # Check if target words are used when sampling context words
        negatives = [t[0] for t in with_negatives if not t[2]]
        negatives_expected = [n for n in range(len(test_tokens)) for _ in range(200)][100:-100]

        self.assertEqual(sum_positives_twos, 2)
        self.assertEqual(sum_negative_twos, 200)
        self.assertEqual(str(negatives), str(negatives_expected))

    def test_03_negative_samples_1(self):  # 1p
        """ Tests whether positive tuples are created correctly."""
        test_tokens = ["a", "rose", "is", "a", "rose"]
        no_negatives = utils.positive_and_negative_cooccurrences(test_tokens, max_distance=1, neg_samples_factor=0,
                                                                 vocab_to_id={"rose": 0, "is": 1, "a": 2})
        no_negatives_expected = {(0, 2, True), (2, 0, True), (1, 0, True), (0, 1, True), (2, 1, True), (1, 2, True),
                                 (0, 2, True), (2, 0, True)}
        self.assertIsNotNone(no_negatives)
        self.assertEqual(set(no_negatives), no_negatives_expected)

    def test_03_negative_samples_2(self):  # 2p
        """ Tests whether return value of positive_and_negative_cooccurrences is of type 'generator'."""
        test_tokens = ["a", "rose", "is", "a", "rose"]
        no_negatives = utils.positive_and_negative_cooccurrences(test_tokens, max_distance=1, neg_samples_factor=0,
                                                                 vocab_to_id={"rose": 0, "is": 1, "a": 2})
        self.assertIsNotNone(no_negatives)
        self.assertIsInstance(no_negatives, types.GeneratorType)

    def test_03_negative_samples_3(self):  # 2p
        """ Tests whether negative tuples are created correctly: Are they chosen randomly from all words? """
        test_tokens = ["0", "0", "2", "3", "4"]
        vocab_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
        pos_neg_cooccurrences = utils.positive_and_negative_cooccurrences(test_tokens, max_distance=1,
                                                                          neg_samples_factor=10,
                                                                          vocab_to_id=vocab_dict)
        self.assertIsNotNone(pos_neg_cooccurrences)
        with_negatives = list(pos_neg_cooccurrences)
        self.assertEqual(len(with_negatives), 88)

        # Count values of negative (sampled) tokens.
        neg_label_distribution = Counter([t[1] for t in with_negatives if t[2] == False])
        # Use chi-squared test, in order to determine whether values are likely to be random.
        expected_distribution = [16, 16, 16, 16, 16]
        p_value = stats.chisquare(list(neg_label_distribution.values()), expected_distribution)[1]
        self.assertGreater(p_value, 0.01)

    def test_03_negative_samples_4(self):  # 2p
        """ Tests whether negative tuples are created correctly: Are they created from all positive tuples? """
        test_tokens = ["0", "0", "2", "3", "4"]
        vocab_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
        pos_negative_cooccurrences = utils.positive_and_negative_cooccurrences(test_tokens, max_distance=1,
                                                                               neg_samples_factor=10,
                                                                               vocab_to_id=vocab_dict)
        self.assertIsNotNone(pos_negative_cooccurrences)
        with_negatives = list(pos_negative_cooccurrences)
        # Count values of negative (sampled) tokens.
        neg_label_contexts = {t[1] for t in with_negatives if t[2] == False}
        self.assertEqual(neg_label_contexts, {0, 1, 2, 3, 4})

    def test_04_skipgram_update_1(self):
        """ Tests whether update on positive tuple is performed correctly."""
        # Check if "positive_and_negative_cooccurrences" isimplemented
        self.assertIsNotNone(utils.positive_and_negative_cooccurrences([], 1, 0, {}))
        sg = skipgram.SkipGram(["b", "a", "c", "b", "c", "c"], window_size=1, neg_samples_factor=0, vocab_size=3,
                               num_dims=5)
        sg.context_word_matrix = np.array([[1, 0, 1], [0, 1, 0], [0, 0, -2]], dtype='float64')
        sg.target_word_matrix = np.array([[-1, 0, -1], [1, 0, 1], [1, 1, 1]], dtype='float64')
        ll = sg.update(context_id=0, target_id=1, label=True, learning_rate=1.0)
        self.assertTrue((sg.context_word_matrix[0] == sg.target_word_matrix[1, :]).all())
        expected_updated = (2 - utils.sigmoid(2)) * np.array([1., 0., 1.], dtype='float64')
        self.assertTrue((sg.context_word_matrix[0] == expected_updated).all())
        self.assertAlmostEqual(ll, -0.127, delta=0.001)

    def test_04_skipgram_update_2(self):
        """ Tests whether update on negative tuple is performed correctly."""
        # Check if "positive_and_negative_cooccurrences" isimplemented
        self.assertIsNotNone(utils.positive_and_negative_cooccurrences([], 1, 0, {}))
        sg = skipgram.SkipGram(["b", "a", "c", "b", "c", "c"], window_size=1, neg_samples_factor=0, vocab_size=3,
                               num_dims=5)
        sg.context_word_matrix = np.array([[1, 0, 1], [0, 1, 0], [0, 0, -2]], dtype='float64')
        sg.target_word_matrix = np.array([[-1, 0, -1], [1, 0, 1], [1, 1, 1]], dtype='float64')
        ll = sg.update(context_id=0, target_id=1, label=False, learning_rate=1.0)
        self.assertTrue((sg.context_word_matrix[0] == sg.target_word_matrix[1]).all())
        expected_updated = (1 - utils.sigmoid(2)) * np.array([1., 0., 1.], dtype='float64')
        self.assertTrue((sg.context_word_matrix[0] == expected_updated).all())
        self.assertAlmostEqual(ll, -2.127, delta=0.001)
