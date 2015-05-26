import numpy as np
import csv


# Read data from features.csv and trainLabels.csv files into X matrix containing features and y vector containing labels
def get_train_data(features_path, labels_path):
    X = np.array(list(csv.reader(open(features_path, "rb"), delimiter=',')), dtype=float)
    # Create y vector from last column, skip first row and convert to int
    y = np.array(list(csv.reader(open(labels_path, "rb"), delimiter=',')))
    y = y[:, 1]
    y = y[1:]
    y = y.astype('int')
    return X, y
