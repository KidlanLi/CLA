from unittest import TestCase
from hw08_neural_networks import get_data, lstm, cnn

class TestNeuralNetworks(TestCase):
    def setUp(self):
        self.texts = [["cat","chases","dog"],
                      ["dog", "chases","cat"],
                      ["dog","and","cat","like","food"],
                      ["cat","sat","on","mat"]]

    def test_01_create_dictionary(self):
        """ Tests whether vocabulary dictionary is created correctly."""
        word_to_id = get_data.create_dictionary(texts=self.texts, vocab_size=3)
        self.assertEqual(word_to_id, {get_data.UNKNOWN_TOKEN:0, "cat":1, "dog":2})

    def test_02_to_ids(self):
        """ Tests whether words are mapped correctly."""
        word_to_id = get_data.create_dictionary(texts=self.texts, vocab_size=3)
        mapped_texts = [get_data.to_ids(t, word_to_id) for t in self.texts]
        mapped_expected = [[1,0,2],[2,0,1],[2,0,1,0,0],[1,0,0,0]]
        self.assertEqual(mapped_expected, mapped_texts)

    def test_03_rnn_training(self):
        """ Tests whether rnns is trained and evaluated correctly."""
        x_train, y_train, x_dev, y_dev, word2id = get_data.nltk_data(vocab_size=lstm.VOCAB_SIZE)
        score, acc, model = lstm.build_and_evaluate_model(x_train, y_train, x_dev, y_dev)
        if not score:
            assert False, 'Solve Task 3: Pad sequences, add layers to the model, compile and fit'
        layer_types = [type(layer).__name__ for layer in model.layers]
        expected_layer_types = ["Embedding", "Bidirectional", "Dense"]
        self.assertEqual(layer_types, expected_layer_types)
        self.assertGreater(acc, 0.67)
        self.assertLess(acc, 0.78)

    def test_04_cnn_training(self):
        """ Tests whether cnn is trained and evaluated correctly."""
        x_train, y_train, x_dev, y_dev, word2id = get_data.nltk_data(vocab_size=cnn.VOCAB_SIZE)
        score, acc, model = cnn.build_and_evaluate_model(x_train, y_train, x_dev, y_dev)
        if not score:
            assert False, 'Solve Task 4: Pad sequences, add layers to the model, compile and fit'
        layer_types = [type(layer).__name__ for layer in model.layers]
        expected_layer_types = ["Embedding", "Conv1D", "GlobalMaxPooling1D", "Dense"]
        self.assertEqual(layer_types, expected_layer_types)
        self.assertGreater(acc, 0.66)
        self.assertLess(acc, 0.77)
