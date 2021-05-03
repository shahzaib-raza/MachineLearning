""" K_Nearest Neighbor For Classification """

import numpy as np
from statistics import mode


class KNearestNeighbor:

    def __init__(self):
        self.y_train = np.array([])
        self.X_train = np.array([])

    def train(self, X, y):
        """
        :param X: N * D dimension metrics of data
        :param y: 1-dimension metrics of labels
        :return: void
        """
        # nearest neighbor simply memorize train data with labels so we jut save them in numpy arrays
        self.X_train = X
        self.y_train = y

    def predict(self, X, K):
        X = np.array(X)
        # we find number of items in test data
        test_size = X.shape[0]
        # we find number of items in train data which we the train method memorized
        train_size = self.X_train.shape[0]
        # we ensure that the size of prediction arrays will be same as of test data size
        y_predict = np.zeros(test_size)

        """ we take each item of test data and subtract it from each item of train data
          take the absolute of subtraction and sum them up in values list.
          The least value of absolute sum of subtraction is most likely to be the same items
          thus we assign the label of that item from train data to the item of test data who 
          are least differ 
          It is the Manhattan distance metric: sum of absolute of differences"""

        for i in range(test_size):
            distances_indexes = []

            # we stor both distance from item in train set and index of that item
            for k in range(train_size):
                distance = np.sum(np.abs(self.X_train[0:][k] - X[0:][i]))
                distances_indexes.append([distance, k])

            # we sort the distance so that minimum difference values and indexes comes first
            distances_indexes.sort()
            # since first index is 0 hence we take k-1 neighbors from sorted difference_index list
            selected = np.array(distances_indexes[:K])
            labels = []

            # we iterate over each index and get the label of train data from that index
            for index in selected[:, 1]:
                labels.append(self.y_train[int(index)])

            # the predicted label will be the most frequent label of the neighbors thus we take mode
            y_predict[i] = mode(labels)

        # return result
        return y_predict
