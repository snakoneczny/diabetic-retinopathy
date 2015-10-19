import numpy as np
import csv


# Read data from csv files into X matrix containing features and y vector containing labels
def get_train_data(features_path, labels_path):
    X = np.array(list(csv.reader(open(features_path, "rb"), delimiter=',')), dtype=float)
    # Create y vector from last column, skip first row and convert to int
    y = np.array(list(csv.reader(open(labels_path, "rb"), delimiter=',')))
    y = y[:, 1]
    y = y[1:]
    y = y.astype('int')
    return X, y


# Balance train data by deleting examples from too big classes
def balance_all(train_X, train_Y):
    unique, counts = np.unique(train_Y, return_counts=True)
    print "Train distribution: %s" % str(counts)
    to_left = np.array([0.32 * counts[0], counts[1], 0.82 * counts[2], counts[3], counts[4]]).astype(int)
    to_delete = counts - to_left
    print "To left: %s" % str(to_left)
    print "To delete: %s" % str(to_delete)

    for i in xrange(5):
        deleted = 0
        while deleted < to_delete[i]:
            idx = np.random.randint(0, len(train_Y))
            if train_Y[idx] == i:
                train_Y = np.delete(train_Y, idx)
                train_X = np.delete(train_X, idx, 0)
                deleted += 1

    unique, counts = np.unique(train_Y, return_counts=True)
    print "Balanced train distribution: %s" % str(counts)

    return train_X, train_Y


# Read data from csv file into X matrix containing features
def get_test_data(path):
    print "Reading test data"
    X = np.array(list(csv.reader(open(path, "rb"), delimiter=',')))
    X = X.astype('float')
    return X


# Read names from csv file into an array
def get_names(path):
    print "Reading names"
    names = np.array(list(csv.reader(open(path, "rb"), delimiter=',')))
    return names


# Based on scores matrix write submission to file
def save_submission(predictions, names, path):
    print "Writing submission: %s" % path
    predictions = predictions.astype('int')
    with open(path, 'wb') as submission:
        # Write description row
        submission.write('image,level\n')
        for i in xrange(len(predictions)):
            # Write new row
            submission.write(names[i][0] + ',' + str(predictions[i]))
            submission.write('\n')
