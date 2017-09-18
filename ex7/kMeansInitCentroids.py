import numpy as np


def kMeansInitCentroids(X, K):
    """returns K initial centroids to be
    used with the K-Means on the dataset X
    """

# You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))

# ====================== YOUR CODE HERE ======================
# Instructions: You should set centroids to randomly chosen examples from
#               the dataset X
#


# =============================================================

    # https://stackoverflow.com/questions/14262654/numpy-get-random-set-of-rows-from-2d-array
    centroids = X[np.random.choice(X.shape[0], K, replace=False), :]
    print(centroids)
    return centroids
