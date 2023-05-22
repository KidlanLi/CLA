from unittest import TestCase

from hw02_paraphrases import paraphrases


class FeaturesTest(TestCase):
    def setUp(self):
        self.text1 = 'kevin durant just got yammed on by carlos delfion'
        self.text2 = 'kevin durant you just got shitted on by carlos delfino'

        self.tokens1 = ['kevin', 'durant', 'just', 'got', 'yammed', 'on', 'by', 'carlos', 'delfino']
        self.tokens2 = ['kevin', 'durant', 'you', 'just', 'got', 'shitted', 'on', 'by', 'carlos', 'delfino']

        self.token_ngrams1 = set(paraphrases.token_ngrams(self.tokens1, 3))
        self.token_ngrams2 = set(paraphrases.token_ngrams(self.tokens2, 3))

        self.character_ngrams1 = set(paraphrases.character_ngrams(self.text1, 3))
        self.character_ngrams2 = set(paraphrases.character_ngrams(self.text2, 3))

        self.t1 = {'ab', 'xy'}
        self.t2 = {'ab', 'cd'}

    # Exercise 1.1
    def test01_token_ngrams(self):
        tokens1_4grams = ['kevin durant just got', 'durant just got yammed', 'just got yammed on',
                        'got yammed on by', 'yammed on by carlos', 'on by carlos delfino']
        tokens2_7grams = ['kevin durant you just got shitted on', 'durant you just got shitted on by',
                        'you just got shitted on by carlos', 'just got shitted on by carlos delfino']
        self.assertEqual(tokens1_4grams, paraphrases.token_ngrams(self.tokens1, 4))
        self.assertEqual(tokens2_7grams, paraphrases.token_ngrams(self.tokens2, 7))

    # Exercise 1.2
    def test02_token_features(self):
        features = dict()
        features[paraphrases.WORD_OVERLAP] = 8
        features[paraphrases.WORD_UNION] = 11
        self.assertEqual(features, paraphrases.token_features(set(self.tokens1), set(self.tokens2)))

    # Exercise 1.3
    def test03_word_ngram_features(self):
        features = dict()
        features[paraphrases.WORD_NGRAM_OVERLAP] = 2
        features[paraphrases.WORD_NGRAM_UNION] = 13
        self.assertEqual(features, paraphrases.word_ngram_features(self.token_ngrams1, self.token_ngrams2))

    # Exercise 1.4
    def test04_character_ngram_features(self):
        features = dict()
        features[paraphrases.CHARACTER_NGRAM_OVERLAP] = 39
        features[paraphrases.CHARACTER_NGRAM_UNION] = 60
        self.assertEqual(features, paraphrases.character_ngram_features(self.character_ngrams1, self.character_ngrams2))

    # Exercise 1.5
    def test05_wordpair_features(self):
        features = dict()
        features['ab#ab'] = 1
        features['ab#cd'] = 1
        features['xy#ab'] = 1
        features['xy#cd'] = 1
        self.assertEqual(features, paraphrases.wordpair_features(self.t1, self.t2))
