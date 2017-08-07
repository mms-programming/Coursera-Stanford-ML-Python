import numpy as np
from numpy.linalg import inv

def normalEqn(X,y):
    """ Computes the closed-form solution to linear regression
       normalEqn(X,y) computes the closed-form solution to linear
       regression using the normal equations.
    """
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#

# ---------------------- Sample Solution ----------------------


# -------------------------------------------------------------

    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html
    # Formula theta = inverted(X transpose * X) * (X transpose * y)
    A = inv(np.dot(X.transpose(), X))
    theta = np.dot(A, np.dot(X.transpose(), y))

    return theta

# ============================================================
