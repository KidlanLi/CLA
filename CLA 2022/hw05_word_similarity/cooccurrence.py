from collections import defaultdict, Counter
from scipy.sparse import lil_matrix
from functools import lru_cache
import math



def vocabulary_from_wordlist(word_list, vocab_size):
    """ Returns set of vocab_size most frequent words from a given list of words.

    >>> v = vocabulary_from_wordlist(['a','rose', 'is', 'a', 'rose', 'colour', 'rose'],2)
    >>> v == {'a', 'rose'}
    True
    >>> v = vocabulary_from_wordlist(['a','rose', 'is', 'a', 'rose'],4)
    >>> v == {'a', 'rose', 'is'}
    True
    """
    freq_dict = Counter(word_list)
    freq_list = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    fw_list = []
    for item in freq_list:
        word,_ = item
        fw_list.append(word)
    if len(fw_list) >= vocab_size:
        return set(fw_list[:vocab_size])
    else:
        return set(fw_list)


def cooccurrences(tokens, n, vocab):
    """
    This takes a list of tokens (representing a text) and returns a dictionary mapping tuples of words
    to their co-occurrence count in windows of n tokens (i.e. the maximum considered distance is n).
    In other words, for each position in the corpus, co-occurrences with n tokens to the left and to the right are
    counted. Only words in a given set of words (the vocabulary) are considered.
    (Note: co-occurrence only holds between words in different positions, not for a position with itself.)

    >>> cooccurrences(["a","rose","is","a","rose"], 2, {"rose", "a"}) == {('rose', 'a'): 3, ('a', 'rose'): 3}
    True
    >>> cooccurrences(["a","rose","is","a","rose"], 1, {"rose", "is"}) == {('rose', 'is'): 1, ('is', 'rose'): 1}
    True
    """

    cooc_dict = defaultdict(int)
    size = n + 1
    for i, f_middle_word in enumerate(tokens):

        # forward (right windows)
        if f_middle_word in vocab:
            f_context = tokens[0 + i + 1:i + size]
            for context_word in f_context:
                if context_word not in vocab:
                    continue
                cooc_dict[(f_middle_word, context_word)] += 1

        # backward (left windows)
        b_middle_word = tokens[-i - 1]

        if b_middle_word not in vocab:
            continue
        b_context = tokens[-i - size:-i - 1]
        for context_word in b_context:
            if context_word not in vocab:
                continue
            cooc_dict[(b_middle_word, context_word)] += 1

    return cooc_dict


def cooc_dict_to_matrix(cooc_dict, vocab):
    """
    This takes a dictionary (word tuples/co-occurrences -> counts) and a vocabulary;
    returns a dictionary mapping each word to an index, as well as
    a Scipy Sparse matrix containing the counts at the index positions.
    >>> d = {('rose', 'is'): 2, ('rose', 'a'): 3, ('a', 'rose'): 3, ('a', 'is'): 4, ('is', 'rose'): 5, ('is', 'a'): 6}
    >>> m, w2id = cooc_dict_to_matrix(d, {'a', 'rose', 'is'})
    >>> w2id == {'is': 1, 'a': 0, 'rose': 2}
    True
    >>> m.toarray()
    array([[0., 4., 3.],
           [6., 0., 5.],
           [3., 2., 0.]])
    >>> m.nnz
    6
    """
    word_to_id = {w: i for i, w in enumerate(sorted(vocab))}
    m = lil_matrix((len(vocab), len(vocab)))
    for (w1, w2), count in cooc_dict.items():
        m[word_to_id[w1], word_to_id[w2]] = count
    # you can use your code from hw04_cooccurrence/cooccurrence.py
    # or
    # use the code provided in cooc_func.nopy
    return m, word_to_id


def ppmi_weight(cooc_matrix):
    """
    This computes a PPMI weighted version of a square matrix with non-negative elements, i.e. a new matrix is returned
    that contains for each cell of the original matrix its PPMI.

    The pointwise information is defined as:
    PMI = log( P(r,c) / (P(r)*P(c)) )
    Where r,c stand for rows and columns of the matrix and:
    P(r,c) = value_of_cell_r_c / sum_of_all_cells
    P(r) = value_of_cells_in_row_r / sum_of_all_cells
    P(c) = value_of_cells_in_column_c / sum_of_all_cells

    The PPMI keeps the positive PMI values, and replaces all negative (or undefined) values with 0.

    >>> m = lil_matrix([[1,2],[3,4]])
    >>> ppmi_weight(m).toarray()
    array([[0.        , 0.10536052],
           [0.06899287, 0.        ]])
    """
    sum_total = cooc_matrix.sum()
    sum_in_col = cooc_matrix.sum(0).A1
    sum_in_row = cooc_matrix.sum(1).A1
    ppmi_matrix = lil_matrix(cooc_matrix.shape)
    rows, cols = cooc_matrix.nonzero()
    for row, col in zip(rows, cols):
        prc = cooc_matrix[row, col]
        pr = sum_in_row[row]
        pc = sum_in_col[col]

        ppmi = math.log(prc) + math.log(sum_total) - math.log(pr) - math.log(pc)

        if ppmi > 0:
            ppmi_matrix[row, col] = ppmi
    return ppmi_matrix
