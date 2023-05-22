from unittest import TestCase
from hw02_paraphrases import paraphrases


class HiddenFeaturesTest(TestCase):

    def setUp(self):
        self.text1 = 'abc de fghi is the best show on tv'
        self.text2 = 'abc de fghi is my favorite show on tv'

        self.tokens1 = ['abc', 'de', 'fghi', 'is', 'the', 'best', 'show', 'on', 'tv']
        self.tokens2 = ['right', 'now', 'abc', 'de', 'fghi', 'is', 'my', 'favorite', 'show', 'on', 'tv',]

        self.token_ngrams1 = set(paraphrases.token_ngrams(self.tokens1, 3))
        self.token_ngrams2 = set(paraphrases.token_ngrams(self.tokens2, 3))

        self.character_ngrams1 = set(paraphrases.character_ngrams(self.text1, 3))
        self.character_ngrams2 = set(paraphrases.character_ngrams(self.text2, 3))

        self.t1 = {'a'}
        self.t2 = {'first', 'second', 'third'}


    # Exercise 1.1
    def test01_token_ngrams_hidden(self):
        tokens1_2grams = ['abc de', 'de fghi', 'fghi is', 'is the', 'the best', 'best show', 'show on', 'on tv']
        tokens2_8grams = ['right now abc de fghi is my favorite', 'now abc de fghi is my favorite show',
                         'abc de fghi is my favorite show on', 'de fghi is my favorite show on tv']
        self.assertEqual(tokens1_2grams, paraphrases.token_ngrams(self.tokens1, 2))
        self.assertEqual(tokens2_8grams, paraphrases.token_ngrams(self.tokens2, 8))
        self.assertEqual(2, len(paraphrases.token_ngrams(self.tokens2, 10)))

    # Exercise 1.2
    def test02_token_features_hidden(self):
        features = dict()
        features[paraphrases.WORD_OVERLAP] = 7
        features[paraphrases.WORD_UNION] = 13
        self.assertEqual(features, paraphrases.token_features(set(self.tokens1), set(self.tokens2)))

    # Exercise 1.3
    def test03_word_ngram_features_hidden(self):
        features = dict()
        features[paraphrases.WORD_NGRAM_OVERLAP] = 3
        features[paraphrases.WORD_NGRAM_UNION] = 13
        self.assertEqual(features, paraphrases.word_ngram_features(self.token_ngrams1, self.token_ngrams2))

    # Exercise 1.4
    def test04_character_ngram_features_hidden(self):
        features = dict()
        features[paraphrases.CHARACTER_NGRAM_OVERLAP] = 22
        features[paraphrases.CHARACTER_NGRAM_UNION] = 45
        self.assertEqual(features, paraphrases.character_ngram_features(self.character_ngrams1, self.character_ngrams2))

    # Exercise 1.5
    def test05_wordpair_features_hidden(self):
        features = dict()
        features['a#first'] = 1
        features['a#second'] = 1
        features['a#third'] = 1
        self.assertEqual(features, paraphrases.wordpair_features(self.t1, self.t2))
        self.assertFalse(paraphrases.wordpair_features({}, self.t2))

