""" Loss Functions and Optimization """

import numpy as np
import math


# simple score function
# score = X.W + b
def score(x, w, b=0):
    """
    :param x: column vector of image tensor
    :param w: weights vector
    :param b: bias vector
    :return: score of classifier
    """
    sc = np.dot(x, w) + b
    return sc


# Multiclass Support Vector Machine(SVM) loss
# loss = sum(max(0, score - true_score + delta)
def svm_loss(scores, ind, delta=1):
    """
    :param ind: index of true class score in scores list
    :param scores: list of scores
    :param delta: margin parameter; default is 1
    :return: loss
    """
    true_class_score = scores[ind]
    new_li = scores.copy()
    new_li.pop(ind)
    loss = sum((max(0, (j - true_class_score + delta))) for j in new_li)
    return loss


# softmax loss
def soft_max_loss(scores, ind):
    """
    :param ind: index of true class score in scores list
    :param scores: list of scores
    :return: loss
    """
    probabilities = []
    un_normalized_prob = [math.e ** i for i in scores]
    for i in un_normalized_prob:
        probabilities.append(i / sum(un_normalized_prob))
    loss = -math.log10(probabilities[ind])
    return loss
