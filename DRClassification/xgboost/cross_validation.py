from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from skll.metrics import kappa
from utils import *
import sys

sys.path.append('D:\\libs\\xgboost\\wrapper')
import xgboost as xgb


# user define objective function, given prediction, return gradient and second order gradient
# this is loglikelihood loss
def logregobj(preds, dtrain):
    labels = dtrain.get_label()

    # TODO: delete
    preds = np.round(preds, 0)
    #print len(labels), labels
    #print len(preds), preds

    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


# user defined evaluation function, return a pair metric_name, result
# NOTE: when you do customized loss function, the default prediction value is margin
# this may make buildin evalution metric not function properly
# for example, we are doing logistic loss, the prediction is score before logistic transformation
# the buildin evaluation error assumes input is after logistic transformation
# Take this in mind when you use the customization, and maybe you need write customized evaluation function
def evalerror(preds, dtrain):
    labels = dtrain.get_label()

    # TODO: delete
    # print 'evalerror'
    # print max(preds)
    preds = np.round(preds, 0)
    # print max(preds)
    # print len(preds), preds

    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'kappa', 1.0 - kappa(labels, preds, weights='quadratic')


# Read data
X, y = get_train_data('../features.csv', '../trainLabels.csv')

# Parameters space creation
params = [[6], [0.3]]
params_space = []
for i in xrange(len(params[0])):
    for j in xrange(len(params[1])):
        params_space.append([params[0][i], params[1][j]])

# Grid search
grid_errors = []
grid_best_iterations = []
for params in params_space:

    # Cross validation
    skf = KFold(len(y), n_folds=4, shuffle=True)  # StratifiedKFold(y, n_folds=4)
    errors = []
    best_iterations = []
    for train, test in skf:
        train_X = X[train]
        train_Y = y[train]
        test_X = X[test]
        test_Y = y[test]
        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_test = xgb.DMatrix(test_X, label=test_Y)

        # Setup parameters
        param = {'silent': 1, 'nthread': 2, 'objective': 'multi:softmax', 'num_class': 5, 'eval_metric': 'mlogloss',
                 'max_depth': params[0], 'eta': params[1]}
        n_rounds = 4000  # Just a big number to trigger early stopping and best iteration

        # Train
        bst = xgb.train(param, xg_train, n_rounds, [(xg_train, 'train'), (xg_test, 'test')], #logregobj, evalerror,
                        early_stopping_rounds=2)

        # Predict
        predictions = bst.predict(xg_test)
        # Get error and best iteration

        # TODO: delete
        # print 'predictions'
        # print max(predictions)
        # predictions = np.round(predictions, 0)
        # print max(predictions)
        # print min(predictions)
        # print len(predictions), predictions

        kappa_score = kappa(test_Y, predictions, weights='quadratic')
        print "Kappa: %f" % kappa_score
        print "Confusion matrix:"
        print confusion_matrix(test_Y, predictions)
        print "Classification report:"
        print classification_report(test_Y, predictions)

        errors.append(kappa_score)
        best_iterations.append(bst.best_iteration)

    # Append new grid error
    grid_errors.append(np.mean(errors))
    grid_best_iterations.append(list(best_iterations))

# Show results
for i in xrange(len(params_space)):
    print "Params: %s, kappa: %f, best iterations: %s, mean: %f" % (
        str(params_space[i]), grid_errors[i], str(grid_best_iterations[i]), np.mean(grid_best_iterations[i]))
