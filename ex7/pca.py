import numpy as np


def pca(X):
    """computes eigenvectors of the covariance matrix of X
      Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """

    # Useful values
    m, n = X.shape

    cov_mat2 = 1.0 / m * np.dot(X.transpose(), X)

    U, S, V = np.linalg.svd(cov_mat2)

    # U = cov_mat2
    # V, S = np.linalg.eig(U)
    #U, S, V = np.linalg.svd(cov_mat2)
    # You need to return the following variables correctly.

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the "svd" function to compute the eigenvectors
    #               and eigenvalues of the covariance matrix.
    #
    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).
    #


# =========================================================================
    return U, np.diag(S), V
