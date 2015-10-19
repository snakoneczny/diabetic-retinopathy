from utils import *
import sys

sys.path.append('D:\\libs\\xgboost\\wrapper')
import xgboost as xgb

# Read data
X_train, y_train = get_train_data('../features_all.csv', '../trainLabels.csv')
X_test = get_test_data("../features_test.csv")
names = get_names("../names.csv")

# Create xbg matrices
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test)

# Setup parameters
param = {'silent': 1, 'nthread': 2, 'objective': 'reg:linear', 'max_depth': 8, 'eta': 0.1}
n_rounds = 30
watchlist = [(xg_train, 'train')]

# Train
bst = xgb.train(param, xg_train, n_rounds, watchlist)

# Predict
predictions = bst.predict(xg_test)

# Clip and round as it is regression
predictions = predictions.clip(0, 4)
predictions = np.round(predictions)

# Save submission to file
save_submission(predictions, names, '../submissions/xgb_reg_linear.csv')
