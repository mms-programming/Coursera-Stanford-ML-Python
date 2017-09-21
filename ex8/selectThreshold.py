import numpy as np
import math

def selectThreshold(yval, pval):
    """
    finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    """

    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    stepsize = (np.max(pval) - np.min(pval)) / 1000.0
    for epsilon in np.arange(np.min(pval),np.max(pval), stepsize):
        predictions = (pval < epsilon)
        predictions = predictions.astype(int)

        TP = np.sum(np.logical_and((predictions == True), (yval == True)).astype(int))
        TN = np.sum(np.logical_and((predictions == False), (yval == False)).astype(int))
        FP = np.sum(np.logical_and((predictions == True), (yval == False)).astype(int))
        FN = np.sum(np.logical_and((predictions == False), (yval == True)).astype(int))
        
        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)

        F1 = 2 * precision * recall / (precision + recall)
        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #
        # Note: You can use predictions = (pval < epsilon) to get a binary vector
        #       of 0's and 1's of the outlier predictions


        # =============================================================

        if F1 > bestF1:
           bestF1 = F1
           bestEpsilon = epsilon

    return bestEpsilon, bestF1
