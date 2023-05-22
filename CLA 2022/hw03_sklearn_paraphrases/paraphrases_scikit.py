import argparse
import sys

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from nltk import word_tokenize

WORD_OVERLAP = "word_overlap"
THREEGRAM_OVERLAP = "threegram_overlap"
TEXT1_LEN = "text1_len"
TEXT2_LEN = "text2_len"
NGRAM1_LEN = "ngram1_len"
NGRAM2_LEN = "ngram2_len"
ZERO=0

def ngrams(s,n):
    """ Returns a list of character n-grams (strings)."""
    return [s[i:i+n] for i in range(len(s)-n+1)]

def token_features(tokens1,tokens2):
    """ Returns a dictionary with token overlap features."""
    features = dict()
    features[WORD_OVERLAP] = intersection_size(tokens1, tokens2)
    features[TEXT1_LEN] = len(tokens1)
    features[TEXT2_LEN] = len(tokens2)
    return features

def ngram_features(ngrams1,ngrams2):
    """ Returns a dictionary with n-gram overlap features."""
    features = dict()
    features[THREEGRAM_OVERLAP] = intersection_size(ngrams1, ngrams2)
    features[NGRAM1_LEN] = len(ngrams1)
    features[NGRAM2_LEN] = len(ngrams2)
    return features

def wordpair_features(tokens1,tokens2):
    """ Returns a dictionary with word-pair features."""
    features = dict()
    for token1 in tokens1:
        for token2 in tokens2:
            features[token1+"#"+token2] = 1
    return features


def paraphrases_to_dataset(filename, vectorizer=None):
    """ Reads a file with tweet pairs, and creates a Dataset for it.
    It can be specified which features should be used."""
    list_of_feature_dicts = []
    list_of_labels = []
    with open(filename, 'r') as myfile:
        for line in myfile:
            parts=line.strip().split("\t")
            text1 = parts[0]
            text2 = parts[1]
            label = parts[2] == "true"
            list_of_labels.append(label)
            tokens1 = set(word_tokenize(text1))
            tokens2 = set(word_tokenize(text2))
            ngrams1 = set(ngrams(text1, 3))
            ngrams2 = set(ngrams(text2, 3))
            features = dict()
            features.update(token_features(tokens1,tokens2))
            features.update(ngram_features(ngrams1,ngrams2))
            features.update(wordpair_features(tokens1,tokens2))
            list_of_feature_dicts.append(features)
    if not vectorizer:
        # TODO Ex.3.1
        vectorizer = DictVectorizer()
    feature_matrix = vectorizer.fit_transform(list_of_feature_dicts)

    # richtiger:
    #    vectorizer = DictVectorizer()
    #    vectorizer.fit(list_of_feature_dicts)
    # feature_matrix = vectorizer.transform(list_of_feature_dicts)

    # TODO: Uncomment the following line and replace the one below:
    return feature_matrix, list_of_labels, vectorizer
    # return None, None , None # <- REPLACE

def readData(trainpath, devpath, testpath):
    """Creates feature matrices from filenames"""
    # TODO Ex 3.2
    train = paraphrases_to_dataset(trainpath)
    train_X = train[0]
    train_Y = train[1]

    dev = paraphrases_to_dataset(devpath) # fehlt Parameter 'vectorizer', sehe unten 'richtiger'
    dev_X = dev[0]
    dev_Y = dev[1]

    test = paraphrases_to_dataset(testpath) # fehlt Parameter 'vectorizer'
    test_X = test[0]
    test_Y = test[1]

    # richtiger:
    # train_X, train_Y, vectorizer = paraphrases_to_dataset(trainpath)
    # dev_X, dev_Y, _ = paraphrases_to_dataset(devpath, vectorizer)
    # test_X, test_Y, _ = paraphrases_to_dataset(testpath, vectorizer)



    # TODO: Uncomment the following line and replace the one below:
    return train_X, train_Y, dev_X, dev_Y, test_X, test_Y
    #return None, None , None , None , None , None # <- REPLACE


def paraphrases_classifier_accuracy(train_file, dev_file, test_file,verbose=False):
    """Trains two classifiers and computes dev accuracies.
    The best classifier is selected and its accuracy on the test set is computed"""
    #return -1 , -1 , -1
    # TODO: uncomment or delete for Ex 3.3
    train_X, train_Y, dev_X, dev_Y, test_X, test_Y = readData(train_file, dev_file, test_file)
    lr_list = [(LogisticRegression(C=0.01, penalty="l2"),'LogisticRegression(C=0.01, penalty="l2")'),
               (LogisticRegression(C=0.1, penalty="l2"),'LogisticRegression(C=0.1, penalty="l2")'),
               (LogisticRegression(C=1.0, penalty="l2"),'LogisticRegression(C=1.0, penalty="l2")'),
               (LogisticRegression(C=0.01, penalty="l1"),'LogisticRegression(C=0.01, penalty="l1")'),
               (LogisticRegression(C=0.1, penalty="l1"),'LogisticRegression(C=0.1, penalty="l1")'),
               (LogisticRegression(C=1.0, penalty="l1"),'LogisticRegression(C=1.0, penalty="l1")'),
               ]
    svc_list = [(LinearSVC(C=0.1) , "LinearSVC(C=0.1)"),
                (LinearSVC(C=1.0) , "LinearSVC(C=1.0)")]
    classifiers_with_names = lr_list + svc_list
    best_dev_acc = 0
    best_classifier = None
    best_classifier_name = None
    for cl, cl_name in classifiers_with_names:

        # TODO Ex 3.3
        cl.fit(train_X, train_Y) # ganz richtig √



        dev_accuracy = accuracy_score(dev_Y, cl.predict(dev_X))
        if verbose:
            print("Classifier: %s - Development Accuracy: %.4f" % (cl_name, dev_accuracy))
        if dev_accuracy > best_dev_acc:
            best_dev_acc = dev_accuracy
            best_classifier = cl
            best_classifier_name = cl_name
    test_accuracy = accuracy_score(test_Y, best_classifier.predict(test_X))
    if verbose:
        print("Best classifier: %s - Test Accuracy: %.4f" % (best_classifier_name, test_accuracy))
    return best_dev_acc, test_accuracy, best_classifier_name

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training', required = True)
    parser.add_argument('-d', '--development', required = True)
    parser.add_argument('-e', '--evaluation', required = True)
    opts = parser.parse_args(argv)
    paraphrases_classifier_accuracy(opts.training, opts.development, opts.evaluation, True)

def intersection_size(i,k):
    """ A strange way to compute set intersection size. Do not use it as a solution for other exercises."""
    return sum([2**ZERO*(j in i) for j in k])

if __name__ == "__main__":
    main(sys.argv[1:])
