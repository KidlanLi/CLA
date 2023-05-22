from unittest import TestCase
from hw08_neural_networks import get_data

class TestNeuralNetworksHidden(TestCase):
    def setUp(self):
        self.texts = [["x","z","y"],
                 ["x", "x","x","z","a"],
                 ["a","b","z"]]

    def test_01_create_dictionary(self):
        """ Tests whether vocabulary dictionary is created correctly."""
        word_to_id = get_data.create_dictionary(texts=self.texts, vocab_size=4)
        self.assertEqual(word_to_id, {get_data.UNKNOWN_TOKEN:0, "x":1, "z":2, "a":3})

    def test_02_to_ids(self):
        """ Tests whether words are mapped correctly."""
        word_to_id = get_data.create_dictionary(texts=self.texts, vocab_size=4)
        mapped_texts = [get_data.to_ids(t, word_to_id) for t in self.texts]
        mapped_expected = [[1,2,0],[1,1,1,2,3],[3,0,2]]
        self.assertEqual(mapped_expected, mapped_texts)
