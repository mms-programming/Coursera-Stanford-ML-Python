import numpy as np
from sklearn import svm

def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

# You need to return the following variables correctly.
    C = 3#1
    sigma = 30 #0.3

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example,
#                   predictions = svmPredict(model, Xval)
#               will return the predictions on the cross validation set.
#
#  Note: You can compute the prediction error using
#        mean(double(predictions ~= yval))
#


# =========================================================================

    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    min_mean = 999
    i = 1
    for c in values:
        for sig in values:

          i = i + 1
          # DO NOT FORGET TO MAKE conversion to from sigma to gamma! 
          gamma = 1.0 / (2.0 * sig ** 2)
          clf = svm.SVC(C=c, kernel='rbf', tol=1e-3, max_iter=200, gamma=gamma)
          model = clf.fit(X, y)

          predictions = model.predict(Xval)
          x = (predictions != yval)
          count_sum = np.sum(x.astype(int))
          count_mean = np.mean(x.astype(int))

          if count_mean < min_mean:

            C = c
            sigma = sig
            min_mean = count_mean

    print(C, sigma)
    return C, sigma
