import argparse
import sys
import os

from hw01_perceptron.utils import DataInstance, Dataset
from hw01_perceptron.perceptron_classifier import PerceptronClassifier


def instances_from_text_files(directory, label):
    """ This returns a generator over data instances in a given directory. """
    instance_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            instance_list.append(DataInstance.from_text_file(directory + "/" + filename, label))
    return instance_list


def main(argv):
    # python3 -m hw01_perceptron_solution.spam_classification -p data/enron/enron1/ham/ -n data/enron/enron1/spam/ -pp data/enron/enron2/ham/ -nn data/enron/enron2/spam/ -ppp data/enron/enron3/ham/ -nnn data/enron/enron3/spam/
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--positive_training_dir', required=True)
    parser.add_argument('-n', '--negative_training_dir', required=True)
    parser.add_argument('-pp', '--positive_dev_dir', required=True)
    parser.add_argument('-nn', '--negative_dev_dir', required=True)
    parser.add_argument('-ppp', '--positive_test_dir', required=True)
    parser.add_argument('-nnn', '--negative_test_dir', required=True)
    opts = parser.parse_args()

    positive_training_instances = instances_from_text_files(opts.positive_training_dir, True)
    negative_training_instances = instances_from_text_files(opts.negative_training_dir, False)
    positive_dev_instances = instances_from_text_files(opts.positive_dev_dir, True)
    negative_dev_instances = instances_from_text_files(opts.negative_dev_dir, False)
    positive_test_instances = instances_from_text_files(opts.positive_test_dir, True)
    negative_test_instances = instances_from_text_files(opts.negative_test_dir, False)

    training_set = Dataset(positive_training_instances + negative_training_instances)
    dev_set = Dataset(positive_dev_instances + negative_dev_instances)
    test_set = Dataset(positive_test_instances + negative_test_instances)

    features = training_set.get_topn_features(1000)

    training_set.set_feature_set(features)
    dev_set.set_feature_set(features)
    test_set.set_feature_set(features)

    perceptron_classifier = PerceptronClassifier.for_dataset(training_set)
    perceptron_classifier.train(training_set, dev_set, 40)

    test_accuracy = perceptron_classifier.prediction_accuracy(test_set)
    test_f_measure = perceptron_classifier.prediction_f_measure(test_set, for_label=False)
    print("Test Accuracy: %.4f" % test_accuracy)
    print("Test F-Measure for SPAM: %.4f" % test_f_measure)
    print("Most frequent sense: %.4f" % test_set.most_frequent_sense_accuracy())
    print("Top features for positive class:")
    print(perceptron_classifier.features_for_class(True))
    print("Top features for negative class:")
    print(perceptron_classifier.features_for_class(False))


if __name__ == "__main__":
    main(sys.argv[1:])
