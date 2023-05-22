from unittest import TestCase
from hw03_sklearn_paraphrases import small_functions as sf
import numpy as np


class TestSmallFunctions(TestCase):
    # Exercise 2.3
    def test03a_trigram_quadragram_vectorizer(self):
        l = ['My name is Bond', 'His name is not Bond', 'I think your name is Bond',
             'I think my name is Bond', 'You think my name is Bond']
        v = sf.trigram_quadragram_vectorizer(l)
        #Damit der Test fail schlägt statt error bei nicht implementierung
        self.assertIsNotNone(v)
        np.testing.assert_equal(['my name is', 'my name is bond', 'name is bond'], v.get_feature_names_out().tolist())

    # Exercise 2.3
    def test03b_trigram_quadragram_vectorizer(self):
        l = ['you are not there', 'you are not there', 'you are not there']
        v = sf.trigram_quadragram_vectorizer(l)
        #Damit der Test fail schlägt statt error bei nicht implementierung
        self.assertIsNotNone(v)
        np.testing.assert_equal(['are not there', 'you are not', 'you are not there'], v.get_feature_names_out().tolist())
