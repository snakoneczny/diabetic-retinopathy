from utils import *
import sys

sys.path.append('D:\\libs\\xgboost\\wrapper')
import xgboost as xgb

# Read data
X_train, y_train = get_train_data('../features_all.csv', '../trainLabels.csv')
X_test = get_test_data("../features_test.csv")
names = get_names("../names.csv")

X_train, y_train = balance_all(X_train, y_train)

# Create xbg matrices
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test)

# Setup parameters
param = {'silent': 1, 'nthread': 2, 'objective': 'multi:softmax', 'eval_metric': 'mlogloss', 'num_class': 5,
         'max_depth': 100, 'eta': 0.08}
n_rounds = 80
watchlist = [(xg_train, 'train')]

# Train
bst = xgb.train(param, xg_train, n_rounds, watchlist)

# Predict
predictions = bst.predict(xg_test)

# Save submission to file
save_submission(predictions, names, '../submissions/xgb_classification.csv')
