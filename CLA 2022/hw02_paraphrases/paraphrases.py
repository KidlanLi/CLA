import argparse
import sys

# from hw02_paraphrases.utils import DataInstance, Dataset, normalized_tokens
# from hw02_paraphrases.perceptron_classifier import PerceptronClassifier

WORD_OVERLAP = "word_overlap"
WORD_UNION = "word_union"

WORD_NGRAM_OVERLAP = "word_ngram_overlap"
WORD_NGRAM_UNION = "word_ngram_union"

CHARACTER_NGRAM_OVERLAP = "character_ngram_overlap"
CHARACTER_NGRAM_UNION = "character_ngram_union"

def character_ngrams(text, n):
    """ Returns a list of character n-grams (strings)."""
    return [text[i:i+n] for i in range(len(text)-n+1)]

def token_ngrams(tokens, n):
    """ Returns a list of lists with n-grams."""
    ngrams = []
    token_address = 0

    for token in tokens[:len(tokens) - n + 1]:
        ngram = token
        for i in range(n-1):
            ngram = ngram + " " + tokens[token_address + i+1]
        token_address += 1
        ngrams.append(ngram)
    # TODO Exercise 1.1
    return ngrams

def token_features(tokens1, tokens2):
    features = dict()

    differ = set(tokens1) & set(tokens2)
    features = {WORD_OVERLAP: len(differ), WORD_UNION: len(tokens1) + len(tokens2) - len(differ)}

    #  TODO Exercise 1.2
    return features

def word_ngram_features(ngrams1, ngrams2):
    features = dict()

    differ = set(ngrams1) & set(ngrams2)
    features = {WORD_NGRAM_OVERLAP: len(differ), WORD_NGRAM_UNION: len(ngrams1) + len(ngrams2) - len(differ)}

    # TODO Exercise 1.3
    return features

def character_ngram_features(ngrams1, ngrams2):
    features = dict()

    differ = set(ngrams1) & set(ngrams2)
    features = {CHARACTER_NGRAM_OVERLAP: len(differ), CHARACTER_NGRAM_UNION: len(ngrams1) + len(ngrams2) - len(differ)}

    # TODO Exercise 1.4
    return features

def wordpair_features(tokens1, tokens2):
    features = dict()

    for token1 in tokens1:
        for token2 in tokens2:
            features[str(token1) + "#" + str(token2)] = 1
    # TODO Exercise 1.5
    return features

def paraphrases_to_dataset(filename, f_token=True, f_w_ngram=True, f_c_ngram=True, f_wordpair=True):
    """
    Reads a file with tweet pairs, and creates a Dataset for it.
    It can be specified which features should be used.
    """
    instance_list = []
    with open(filename, 'r') as myfile:
        for line in myfile:
            parts=line.strip().split("\t")
            text1 = parts[0]
            text2 = parts[1]
            label = parts[2] == "true"
            tokens1 = normalized_tokens(text1)  # removed sets
            tokens2 = normalized_tokens(text2)

            features = dict()
            if f_token:
                features.update(token_features(set(tokens1), set(tokens2)))
            if f_w_ngram:
                token_ngrams1 = set(token_ngrams(tokens1, 3))
                token_ngrams2 = set(token_ngrams(tokens2, 3))
                features.update(word_ngram_features(token_ngrams1, token_ngrams2))
            if f_c_ngram:
                character_ngrams1 = set(character_ngrams(text1, 3))
                character_ngrams2 = set(character_ngrams(text2, 3))
                features.update(character_ngram_features(character_ngrams1, character_ngrams2))
            if f_wordpair:
                features.update(wordpair_features(tokens1,tokens2))
            inst = DataInstance(features, label)
            instance_list.append(inst)
    return Dataset(instance_list)

def feature_comparison(trainpath, devpath, print_output=False):
    """
    Returns development accuracies for the following feature combinations:
     - Only one feature activated at a time (4 times)
     - wordpair, character ngram features
     - wordpair, character ngram, word ngram features
     - wordpair, character ngram, word ngram features, token features
    """
    devaccs = []
    if print_output: print("---------------")
    # Create tuple list with signatures for each feature activated one at a time + stacking feature combinations
    for (signature, explanation) in \
            [([False, False, False, True], "Only wordpair features"),
             ([False, False, True,  False], "Only character ngram features"),
             ([False, True,  False, False], "Only word ngram features"),
             ([True,  False, False, False], "Only token features"),
             ([False, False, True,  True], "wordpair and character ngram features"),
             ([False, True,  True,  True], "wordpair, character ngram and word ngram features"),
             ([True,  True,  True,  True], "wordpair, character ngram, word ngram and token features")]:

        if print_output: print(explanation)
        train_set = paraphrases_to_dataset(trainpath, *signature)
        dev_set = paraphrases_to_dataset(devpath, *signature)
        classifier = PerceptronClassifier.for_dataset(train_set)
        classifier.train(train_set, dev_set, 20, verbose=False)
        dev_accuracy = classifier.prediction_accuracy(dev_set)
        if print_output: print("Dev acc:", dev_accuracy)
        if print_output: print("---------------")
        devaccs.append(dev_accuracy)
    return devaccs

def main(argv):
    """ Trains and evaluates the classifier on data in Semeval-2015 data."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training', required=True)
    parser.add_argument('-d', '--development', required=True)
    parser.add_argument('-e', '--evaluation', required=True)
    parser.add_argument('-fc', '--featurecomparison', action='store_true')
    opts = parser.parse_args(argv)

    train_set = paraphrases_to_dataset(opts.training)
    dev_set = paraphrases_to_dataset(opts.development)
    test_set = paraphrases_to_dataset(opts.evaluation)

    mfs_dev = dev_set.most_frequent_sense_accuracy()
    print("Most frequent sense (dev): %s " % mfs_dev)
    if opts.featurecomparison:
        print("\nFEATURE COMPARISON MODE")
        feature_comparison(opts.training, opts.development, print_output=True)
    else:
        classifier = PerceptronClassifier.for_dataset(train_set)
        classifier.train(train_set, dev_set, 20)
        test_accuracy = classifier.prediction_accuracy(test_set)
        mfs_test = test_set.most_frequent_sense_accuracy()
        print("Most frequent sense (test):", mfs_test)
        print("Test Accuracy: %.4f" % (test_accuracy))

if __name__ == "__main__":
    main(sys.argv[1:])
