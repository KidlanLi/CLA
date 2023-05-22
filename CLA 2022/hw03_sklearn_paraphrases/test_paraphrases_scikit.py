from unittest import TestCase
from hw03_sklearn_paraphrases import paraphrases_scikit as ps
import numpy as np


class Test_paraphrases_scikit(TestCase):
    def setUp(self):
        self.trainpath_small = "hw03_sklearn_paraphrases/unittest-toydata/train.txt"
        self.devpath_small = "hw03_sklearn_paraphrases/unittest-toydata/dev.txt"
        self.testpath_small = "hw03_sklearn_paraphrases/unittest-toydata/test.txt"
        self.trainpath = "data/paraphrases/train.txt"
        self.devpath = "data/paraphrases/dev.txt"
        self.testpath = "data/paraphrases/test.txt"

    def test_01_paraphrases_to_dataset(self):
        """Calls paraphrases_to_dataset on the toy data
        and checks if the retured matrices look like expected"""
        train_X, train_Y, vectorizer= ps.paraphrases_to_dataset(self.trainpath_small)
        self.assertIsNotNone(train_X)
        x= [[1.,  0.,  1.,  1.,  0.,  2.,  1.,  0.,  1.],
            [0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.]]
        self.assertCountEqual(train_X.toarray().tolist() ,x)

        dev_X, dev_Y, _= ps.paraphrases_to_dataset(self.devpath_small, vectorizer)
        x = [[ 1.,  0.,  0.,  1.,  0.,  2.,  1.,  0.,  1.],
             [ 0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.]]
        self.assertCountEqual(dev_X.toarray().tolist(), x)
        self.assertCountEqual(dev_Y, [False, False])

    def test_02_read_data(self):
        """Calls readData on the toy data
        and checks if the retured matrices look like expected"""
        train_X, train_Y, dev_X, dev_Y, test_X, test_Y = ps.readData(self.trainpath_small, self.devpath_small, self.testpath_small)
        self.assertIsNotNone(train_X)
        self.assertIsNotNone(dev_X)
        self.assertIsNotNone(test_Y)
        x = [[ 1.,  0.,  1.,  1.,  0.,  2.,  1.,  0.,  1.],
            [ 0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.]]
        self.assertCountEqual(train_X.toarray().tolist() ,x)
        x = [[ 1.,  0.,  0.,  1.,  0.,  2.,  1.,  0.,  1.],
             [ 0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.]]
        self.assertCountEqual(dev_X.toarray().tolist(), x)
        self.assertCountEqual(test_Y, [False, False])

    def test_03_training(self):
        """Checks if the classifier returned by paraphrases_classifier_accuracy
        achieves the expected scores. (Uses the 'real' data and not the toy data)"""
        dev_acc, test_acc, cl_name = ps.paraphrases_classifier_accuracy(self.trainpath, self.devpath, self.testpath)
        self.assertAlmostEqual(dev_acc, 0.72, delta=0.05)
        self.assertAlmostEqual(test_acc, 0.86, delta=0.05)
