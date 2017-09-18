import numpy as np


def findClosestCentroids(X, centroids):
    """returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
    """

# Set K
    K = len(centroids)

# You need to return the following variables correctly.
    idx = np.zeros(X.shape[0])

    val = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Go over every example, find its closest centroid, and store
#               the index inside idx at the appropriate location.
#               Concretely, idx(i) should contain the index of the centroid
#               closest to example i. Hence, it should be a value in the
#               range 1..K
#
# Note: You can use a for-loop over the examples to compute this.


# =============================================================
    for i,x in enumerate(X):
      idx[i] = int(findCloseCentroid(x, centroids))

    return val, idx


def findCloseCentroid(x, centroids):
  # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
  # https://stackoverflow.com/questions/8079061/function-application-over-numpys-matrix-row-column
  dist = [np.linalg.norm(x-centroid) for centroid in centroids]
  return np.argmin(dist)
