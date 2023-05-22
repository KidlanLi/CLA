import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def square_roots(start, end, length):
    """
    Returns a 1d numpy array of the specified length, containing the square roots of equi-distant input values
    between start and end (both included).

    #>>> square_roots(4,9,3)
    array([2.        , 2.54950976, 3.        ])
    """
    linarray = np.linspace(start, end, num=length) # TODO: Exercise 2.1
    return np.sqrt(linarray)

def odd_ones_squared(rows, cols):
    """
    Returns a 2d numpy array with shape (rows, cols). The matrix cells contain increasing numbers,
    where all odd numbers are squared.

    #>>> odd_ones_squared(3,5)
    array([[  0,   1,   2,   9,   4],
           [ 25,   6,  49,   8,  81],
           [ 10, 121,  12, 169,  14]])
    """
    matrix = np.arange(rows*cols).reshape(rows, cols)
    #matr_array = np.array(matrix)
    #print(matr_array)
    np.putmask(matrix, matrix % 2 == 1, matrix**2)
    return matrix
