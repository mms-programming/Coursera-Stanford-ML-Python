from numpy import e
import numpy as np

def sigmoid(z):
    """computes the sigmoid of z."""

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar).

# =============================================================
    return 1 / (1 + np.exp(-z))
