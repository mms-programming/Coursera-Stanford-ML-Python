import numpy as np

def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear
       regression to fit the data points in X and y
    """
    m = y.size
    J = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.


# =========================================================================

    # Use ndarrays
    h_theta = np.dot(X, theta)
    J = sum(np.square(h_theta - y)) / (2 * m )

    return J

'''
    # Use np matrices instead
    X_mat = np.matrix(X)
    y_mat = np.matrix(y)
    theta_mat = np.matrix(theta)

    # X_mat is (97, 2), theta_mat starts off as (1, 2), y starts as (1, 97)
    h_theta_mat = X_mat * theta_mat.transpose()

    J = sum(np.square(h_theta_mat - y_mat.transpose())) / (2 * m)
'''
