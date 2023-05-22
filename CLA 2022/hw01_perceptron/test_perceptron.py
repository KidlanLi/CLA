from unittest import TestCase

from hw01_perceptron.perceptron_classifier import PerceptronClassifier
from hw01_perceptron.utils import DataInstance, Dataset, dot

# Ham mails: True (positive data)
# Spam mails: False (negative data)
train_inst_1 = [DataInstance.from_list_of_feature_occurrences(sample[0], sample[1]) for sample in
                [("deal lottery lottery lottery".split(" "), False),
                 ("lottery lottery".split(" "), False),
                 ("deal".split(" "), True),
                 ("deal deal deal lottery".split(" "), True)]]

train_inst_2 = [DataInstance.from_list_of_feature_occurrences(sample[0], sample[1]) for sample in
                [("deal lottery lottery lottery".split(" "), False),
                 ("lottery lottery".split(" "), False),
                 ("deal".split(" "), True),
                 ("green eggs".split(" "), True)]]

# This is used to test accuracy. A correctly implemented classifier should NOT reach 100% here.
dev_inst = [DataInstance.from_list_of_feature_occurrences(sample[0], sample[1]) for sample in
            [("deal deal deal".split(" "), True),
             ("deal deal lottery".split(" "), True),
             ("deal lottery deal lottery deal".split(" "), False)]]

# A correctly implemented classifier will reach 100% accuracy here
pred_inst_1 = [DataInstance.from_list_of_feature_occurrences(sample[0], sample[1]) for sample in
               [("deal".split(" "), True),
                ("lottery".split(" "), False),
                ("deal deal lottery".split(" "), True),
                ("deal deal lottery lottery".split(" "), False),
                ("unknown".split(" "), False)]]

pred_inst_2 = [DataInstance.from_list_of_feature_occurrences(sample[0], sample[1]) for sample in
               [("deal".split(" "), True),
                ("lottery".split(" "), False),
                ("deal lottery".split(" "), False),
                ("deal deal lottery".split(" "), False),
                ("lottery lottery lottery".split(" "), False)]]

# No update needed
no_update_inst = [DataInstance.from_list_of_feature_occurrences(sample[0], sample[1]) for sample in
                  [("deal deal lottery".split(" "), True),
                   ("lottery".split(" "), False),
                   ("deal lottery".split(" "), False)]]

# Do update
do_update_inst = [DataInstance.from_list_of_feature_occurrences(sample[0], sample[1]) for sample in
                  [("deal".split(" "), False),
                   ("lottery".split(" "), True),
                   ("deal deal lottery".split(" "), False)]]


class PerceptronClassifierTest(TestCase):

    def setUp(self):
        self.small_dataset_train_1 = Dataset(train_inst_1)
        self.small_dataset_train_2 = Dataset(train_inst_2)
        self.small_dataset_dev = Dataset(dev_inst)
        self.small_dataset_pred_test_1 = Dataset(pred_inst_1)
        self.small_dataset_pred_test_2 = Dataset(pred_inst_2)
        self.small_instance_list_no_update = no_update_inst
        self.small_instance_list_do_update = do_update_inst

    def test01_dot_product_01(self):
        """Checks if dot product is correctly calculated"""
        dictA = {'Car': 42, 'Apple': 1, 'Banana': 25}
        dictB = {'Apple': 10, 'House': 52, 'Car': 24}
        expected_value = 1072
        self.assertEqual(dot(dictA, dictB), expected_value)

    def test02_creating_data_instance_01(self):
        """Checking if Data instance created correctly"""
        feature_list = ['the', 'cat', 'sat', 'on', 'the', 'mat']
        label = 'spam'
        instance = DataInstance.from_list_of_feature_occurrences(feature_list, label)
        self.assertEqual(instance.label, label)
        self.assertEqual(instance.feature_counts, {'the': 2, 'cat': 1, 'sat': 1, 'on': 1, 'mat': 1})

    def test03_most_frequent_features_01(self):
        ''' Checking if most n frequent words are retrieved correctly '''
        self.assertEqual(self.small_dataset_train_2.get_topn_features(2), {'deal', 'lottery'})
        self.assertEqual(self.small_dataset_dev.get_topn_features(1), {'deal'})

    def test04_filtering_features_01(self):
        '''Checking if feature counts only contains features from the filter'''
        filter = {'lottery', 'eggs', 'Banana'}
        self.small_dataset_train_2.set_feature_set(filter)
        self.assertEqual(self.small_dataset_train_2.instance_list[0].feature_counts.keys(), {'lottery'})
        self.assertEqual(self.small_dataset_train_2.instance_list[3].feature_counts.keys(), {'eggs'})

    def test05_most_frequent_baseline_01(self):
        '''Checking if the base line accuracy is calculated correctly'''
        self.assertEqual(self.small_dataset_pred_test_1.most_frequent_sense_accuracy(), 0.6)

    def test_06_update_01(self):
        """Verify that the perceptron update is performed correctly."""
        classifier = PerceptronClassifier({'deal': 1, 'lottery': -1})
        # Test document: ("deal", "doc25", False)
        classifier.update(self.small_instance_list_do_update[0])
        expected_weigths = {'deal': 0, 'lottery': -1}
        self.assertEqual(classifier.weights, expected_weigths)

        classifier = PerceptronClassifier({'deal': 1, 'lottery': -1})
        # Test document: ("lottery", "doc26", True),
        do_update = classifier.update(self.small_instance_list_no_update[0])
        self.assertEqual(False, do_update)

    def test_08_f1_measure_01(self):
        """Checking if the F1 measure is calculated correctly"""
        pc = PerceptronClassifier.for_dataset(self.small_dataset_train_1)
        self.assertAlmostEqual(pc.prediction_f_measure(self.small_dataset_train_1, False), 0.666666666)
