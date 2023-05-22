from unittest import TestCase
from hw03_sklearn_paraphrases import small_functions as sf
import numpy as np

#Solution
class TestSmallFunctions(TestCase):
    # Exercise 2.3
    def test03a_trigram_quadragram_vectorizer_hidden(self):
        l = ['one two three ' *4] + ['two three one ' * 5] + ['three one two ' * 5]
        v = sf.trigram_quadragram_vectorizer(l)
        self.assertIsNotNone(v)
        np.testing.assert_equal(6, len(v.get_feature_names()))

    # Exercise 2.3
    def test03b_trigram_quadragram_vectorizer_hidden(self):
        l = ['cat ' * 4, 'the cat cat cat', 'cat ' * 4 + 'dog']
        v = sf.trigram_quadragram_vectorizer(l)
        self.assertIsNotNone(v)
        np.testing.assert_equal(['cat cat cat'], v.get_feature_names())
