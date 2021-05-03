""" Nearest Neighbor For Classification """

import numpy as np


class NearestNeighbor:

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

    def predict(self, X):
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
            values = []

            for k in range(train_size):
                distance = np.sum(np.abs(self.X_train[0:][k] - X[0:][i]))
                values.append(distance)
            lowest_index = np.argmin(values)
            y_predict[i] = self.y_train[lowest_index]

        # return result
        return y_predict
