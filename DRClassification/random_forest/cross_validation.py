from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from skll.metrics import kappa

from xgboost.utils import *




# Read data
X, y = get_train_data('../features_all.csv', '../trainLabels.csv')

# Parameters space creation
params_space = [[200]]

# Grid search
grid_errors = []
for params in params_space:

    # Cross validation
    skf = StratifiedKFold(y, n_folds=8)

    errors = []
    for train, test in skf:

        clf = RandomForest(n_estimators=params[0], n_jobs=2)
        clf.fit(X[train], y[train])
        predictions = clf.predict(X[test])

        kappa_score = kappa(y[test], predictions, weights='quadratic')
        print "Kappa: %f" % kappa_score
        print "Confusion matrix:"
        print confusion_matrix(y[test], predictions)
        print "Classification report:"
        print classification_report(y[test], predictions)

        errors.append(kappa_score)

    grid_errors.append(np.mean(errors))

# Show results
print "Kappa: " + str(grid_errors)
