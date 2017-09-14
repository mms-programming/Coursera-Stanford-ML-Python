import numpy as np


def gaussianKernel(x1, x2, sigma):
    """returns a gaussian kernel between x1 and x2
    and returns the value in sim
    """

# Ensure that x1 and x2 are column vectors
#     x1 = x1.ravel()
#     x2 = x2.ravel()

# You need to return the following variables correctly.
    sim = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the similarity between x1
#               and x2 computed using a Gaussian kernel with bandwidth
#               sigma
#
#


# =============================================================

    # Keys to remember
    # 1. np.power for exponents
    # 2. float when it comes to multiplication and divison
    sim = np.exp(-1.0 * np.sum(np.square(x1-x2)) / (2 * np.power(sigma, 2)))

    return sim
