import numpy as np


def zeros(n):
    """"
    Creates a list of full zeros just like in the Matlab.
    """
    return [0] * n


def normalization(vector):
    """
    Classical 0-1 normalization.
    :param vector:
    :return: normalized vector.
    """
    return [((x - min(vector)) / (max(vector) - min(vector))) for x in vector]


def add_dimension(vector):
    """ If the ndarray sized (n,) adds second dimension
    makes all (n, 1) sized tuple
    :param
        vector: Any ndarray sized (n,)
    :return
        vector : ndarray sized (n,1)
    """
    return np.array([[node] for node in vector])