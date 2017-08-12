from ex2.sigmoid import sigmoid
import numpy as np

def sigmoidGradient(z):
    """computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector. In particular, if z is a vector or matrix, you should return
    the gradient for each element."""

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of the sigmoid function evaluated at
#               each value of z (z can be a matrix, vector or scalar).


# =============================================================
    # Note for matrix you need do elementwise multiplication
    return np.multiply(sigmoid(z), 1 - sigmoid(z))
