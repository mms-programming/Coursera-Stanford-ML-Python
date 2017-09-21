import numpy as np


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda):
    """returns the cost and gradient for the
    """

    # Unfold the U and W matrices from params
    X = np.array(params[:num_movies*num_features]).reshape(num_features, num_movies).T.copy()
    Theta = np.array(params[num_movies*num_features:]).reshape(num_features, num_users).T.copy()

    Theta_X = np.dot(X, Theta.T)
    R_ints = R.astype(int)
    #Matrix dot product is *
    actual_results = Theta_X * R_ints
    # Need to fix the issue that Y is non-zero for movies that a user has not rated
    Y_results = Y * R_ints

    # You need to return the following values correctly
    computation = actual_results - Y_results
    J = np.sum(np.sum((computation) ** 2)) / 2

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    X_grad = np.dot(computation, Theta) + Lambda * X
    Theta_grad = np.dot(computation.T, X) + Lambda * Theta 

    J = J + Lambda / 2.0 * (np.sum(np.sum(np.power(Theta, 2))) +
                             np.sum(np.sum(np.power(X, 2))))




    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies  x num_features matrix of movie features
    #        Theta - num_users  x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R(i, j) = 1 if the
    #            i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly:
    #
    #        X_grad - num_movies x num_features matrix, containing the
    #                 partial derivatives w.r.t. to each element of X
    #        Theta_grad - num_users x num_features matrix, containing the
    #                     partial derivatives w.r.t. to each element of Theta


    # =============================================================

    grad = np.hstack((X_grad.T.flatten(),Theta_grad.T.flatten()))

    return J, grad
