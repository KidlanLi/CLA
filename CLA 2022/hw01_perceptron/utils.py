import random
from nltk import word_tokenize


def dot(dictA, dictB):
    listA = dictA.values()
    listB = dictB.values()
    scalar = sum([x * y for x, y in zip(listA, listB)])
    return scalar  # TODO: Ex. 2: return vector product between features vectors represented by dictA and dictB.


def normalized_tokens(text):
    return [token.lower() for token in word_tokenize(text)]


class DataInstance:
    def __init__(self, feature_counts, label):
        """ A data instance consists of a dictionary with feature counts (string -> int) and a label (True or False)."""
        self.feature_counts = feature_counts
        self.label = label

    @classmethod
    def from_list_of_feature_occurrences(cls, feature_list, label):
        """ Creates feature counts for all features in the list."""
        feature_counts = dict({item: feature_list.count(item) for item in set(feature_list)})
        # TODO: Ex. 3: create a dictionary that contains for each feature in the list the count how often it occurs.
        return cls(feature_counts, label)

    @classmethod
    def from_text_file(cls, filename, label):
        with open(filename, 'r') as myfile:
            token_list = normalized_tokens(myfile.read().strip())
        return cls.from_list_of_feature_occurrences(token_list, label)


class Dataset:
    def __init__(self, instance_list):
        """ A data set is defined by a list of instances """
        self.instance_list = instance_list
        self.feature_set = set.union(*[set(inst.feature_counts.keys()) for inst in instance_list])

    def get_topn_features(self, n):
        """ This returns a set with the n most frequently occurring features (i.e. the features that are contained in most instances)."""
        new_dict = dict()

        for ele in self.feature_set:
            frequency = 0
            for inst in self.instance_list:
                if ele in inst.feature_counts.keys():
                    frequency = frequency + inst.feature_counts[ele]
            new_dict[ele] = frequency
        features_frequency = sorted(new_dict.items(), key=lambda x: x[1], reverse=True)
        return set(features_frequency[i][0] for i in
                   range(n))  # TODO: Ex. 4: Return set of n features that occur in most instances.

    def set_feature_set(self, feature_set):
        """
        This restrics the self.feature_set. Only features in the specified set all retained. All other feature are removed
        from all instances in the dataset AND from the self.feature set."""
        self.feature_set.intersection(feature_set)
        for inst in self.instance_list:
            diff_set = set(inst.feature_counts.keys()).difference(feature_set)
            for ele in diff_set:
                del inst.feature_counts[ele]
        # TODO: Ex. 5: Filter features according to feature set.

    def most_frequent_sense_accuracy(self):
        """ Computes the accuracy of always predicting the overall most frequent label for all instances in the dataset. """
        count_true = count_false = 0
        for inst in self.instance_list:
            if inst.label:
                count_true = count_true + 1
            else:
                count_false = count_false + 1

        if count_true > count_false:
            result = count_true / (count_true + count_false)
        else:
            result = count_false / (count_true + count_false)
        return result
        # TODO: Ex. 6: Return accuracy of always predicting most frequent label in data set.

    def shuffle(self):
        """ Shuffles the dataset. Beneficial for some learning algorithms."""
        random.shuffle(self.instance_list)
