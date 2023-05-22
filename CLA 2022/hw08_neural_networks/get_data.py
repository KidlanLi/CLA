import collections
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize

random.seed(111)

UNKNOWN_TOKEN = "<unk>"

def create_dictionary(texts, vocab_size):
    """
    Creates a dictionary that maps words to ids. More frequent words have lower ids.
    The dictionary contains at the vocab_size-1 most frequent words (and a placeholder '<unk>' for unknown words).
    The place holder has the id 0.

    :param texts: list of word lists
    :param vocab_size

    :return A dictionary that maps words to their id.
    """
    counter = collections.Counter()
    for tokens in texts:
        counter.update(tokens)
    vocab = [w for w, c in counter.most_common(vocab_size - 1)]
    """w2id = {i: w for w, i in enumerate(vocab, 1)}
    w2id["<unk>"] = 0"""

    w2id = {w: (i + 1) for i, w in enumerate(vocab)}
    w2id[UNKNOWN_TOKEN] = 0
    return w2id  # TODO: Exercise 1.

def to_ids(words, dictionary):
    """
    Takes a list of words and converts them to ids using the word2id dictionary.
    :param words: list of tokens
    :param dictionary: maps tokens to their id

    :return list of ids for each token (placeholder '<unk>' for unknown tokens)
    """
    ids = []
    for word in words:
        ids.append(dictionary.get(word, dictionary[UNKNOWN_TOKEN]))
    return ids  # TODO: Exercise 2.


def nltk_data(n_texts_train=1500, n_texts_dev=500, vocab_size=10000):
    """
    Reads texts from the nltk movie_reviews corpus. A word2id dictionary is 
    created and the words in the texts are substituted with their numbers. Training
    and Development data is returned, together with labels and the word2id dictionary.
 
    :param n_texts_train: the number of reviews that will form the training data
    :param n_texts_dev: the number of reviews that will form the development data
    :param vocab_size: the maximum size of the vocabulary.

    :return list texts_train: A list containing lists of wordids corresponding to 
    training texts.
    :return list texts_dev: A list containing lists of wordids corresponding to 
    development texts.
    :return labels_train: A list containing the labels (0 or 1) for the corresponding
    text entry in texts_train
    :return labels_dev: A ilst containing the labels (0 or 1) for the corresponding
    text entry in texts_dev
    :return word2id: The dictionary obtained from the training texts that maps each
    seen word to an id.
    """
    all_ids = movie_reviews.fileids()
    if (n_texts_train+n_texts_dev>len(all_ids)):
        print ("Error: There are only",len(all_ids), "texts in the movie_reviews corpus. Training with all of those sentences.")
        n_texts_train=1500
        n_texts_dev=500
    posids = movie_reviews.fileids('pos')
    random.shuffle(all_ids)

    texts_train=[]
    labels_train=[]
    texts_dev=[]
    labels_dev=[]

    for i in range(n_texts_train):
        text = movie_reviews.raw(fileids=[all_ids[i]])
        tokens = [word.lower() for word in word_tokenize(text)]
        texts_train.append(tokens)
        if all_ids[i] in posids:       
            labels_train.append(1)
        else:
            labels_train.append(0)

    for i in range(n_texts_train, n_texts_train+n_texts_dev):
        text = movie_reviews.raw(fileids=[all_ids[i]])
        tokens = [word.lower() for word in word_tokenize(text)]
        texts_dev.append(tokens)
        if all_ids[i] in posids:
            labels_dev.append(1)
        else:
            labels_dev.append(0)

    word2id=create_dictionary(texts_train, vocab_size)
    texts_train = [to_ids(s,word2id) for s in texts_train]
    texts_dev = [to_ids(s,word2id) for s in texts_dev]
    return (texts_train, labels_train, texts_dev, labels_dev, word2id)

