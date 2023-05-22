import random, math
import numpy as np
from hw06_skipgram import utils

class SkipGram:
    """ Class for training skipgram embeddings from a corpus. """
    def __init__(self, tokens, window_size=1, neg_samples_factor=10, vocab_size=10000, num_dims=50):
        """
        Creates an object for training skipgram embeddings from a corpus

        :param tokens: List of strings, the corpus.
        :param window_size: Maximum distance of context words.
        :param neg_samples_factor: Number of sampled negative tuples for each positive tuple
        :param vocab_size: Dictionary (string to int) mapping each word to its id (=row in embedding matrizes).
        :param num_dims: Number of dimensions used for embedding matrizes.
        """
        self.word_to_id = utils.vocabulary_to_id_for_wordlist(tokens, vocab_size)
        self.pos_neg_list = list(utils.positive_and_negative_cooccurrences(tokens, window_size, neg_samples_factor, self.word_to_id))
        rows = len(self.word_to_id)
        self.target_word_matrix = 0.1 * np.random.randn(rows, num_dims)
        self.context_word_matrix = 0.1 * np.random.randn(rows, num_dims)

    def update(self, target_id, context_id, label, learning_rate): # TODO: Exercise 4.
        """
        Performs a gradient update for one instance.

        :param target_id: Row number of target word to predict.
        :param context_id: Row number of context word.
        :param label: Indicator whether pair is co-occurrence from corpus.
        :param learning_rate: Multiplier for the magnitude of gradient step.
        :return: Log-likelihood of of current example *before* the update.
        """
        ctxt_vec = self.context_word_matrix[context_id]
        tgt_vec = self.target_word_matrix[target_id]
        # TODO: Replace by probability that pair belongs to positive category.
        prob_pos = utils.sigmoid(ctxt_vec.dot(tgt_vec))  # no difference between ctxt_vec.dot(tgt_vec.T)
        # TODO: update context_word_matrix[context_id] using gradient and learning rate
        # TODO: update target_word_matrix[target_id] using gradient and learning rate
        ctxt2 = np.copy(ctxt_vec)
        tgt2 = np.copy(tgt_vec)

        factor = (label - prob_pos) * learning_rate
        self.context_word_matrix[context_id] += factor * tgt2
        self.target_word_matrix[target_id] += factor * ctxt2
        return math.log(prob_pos) if label else math.log(1 - prob_pos)


        return math.log(prob_pos) if label else math.log(1 - prob_pos)

    def train_iter(self, learning_rate=0.1):
        """
        Performs gradient updates for all positive and negative tuples in the training set.

        :param learning_rate: Multiplier for the magnitude of gradient step.
        """
        random.shuffle(self.pos_neg_list)
        ll = 0
        for tgt_id, ctxt_id, lbl in self.pos_neg_list:
            ll += self.update(tgt_id, ctxt_id, lbl, learning_rate)
        print("Log-likelihood: %.4f" % (ll))

    def toDenseSimilarityMatrix(self):
        """
        This creates a DenseSimilarityMatrix from the target word embedding matrix for similarity computations.

        :return: The DenseSimilarityMatrix object.
        """
        return utils.DenseSimilarityMatrix(self.target_word_matrix, self.word_to_id)
