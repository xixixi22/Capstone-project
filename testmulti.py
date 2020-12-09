import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV

def correlation(label_test, label_predict, correlation_type):
    if correlation_type == 'pearson':
        corr = pearsonr
    elif correlation_type == 'spearman':
        corr = spearmanr
    else:
        raise ValueError("Unknown correlation type: %s" % correlation_type)
    score = []
    for lb_test, lb_predict in zip(label_test, label_predict):
        score.append(corr(lb_test, lb_predict)[0])
    return np.mean(score), score



def gridSearch(base_model, random_grid):
    random = GridSearchCV(MultiOutputRegressor(base_model), param_grid=random_grid,
                          cv=3, n_jobs=-1)

    random.fit(train_X, train_y)
    print(random.best_params_)
    best_random = random.best_estimator_
    pred_y_test = best_random.predict(test_X)
    print_scores(test_y_array, pred_y_test)
    pred_y_dev = best_random.predict(dev_X)
    print_scores(dev_y_array, pred_y_dev)
    
train = pd.DataFrame(pd.read_csv("./data/train.csv"))
test = pd.DataFrame(pd.read_csv("./data/test.csv"))
dev = pd.DataFrame(pd.read_csv("./dev.csv"))

train_X,train_y=train[train.columns[978:]],train[train.columns[:978]]
test_X,test_y=test[test.columns[978:]],test[test.columns[:978]]
dev_X,dev_y=dev[dev.columns[978:]],dev[dev.columns[:978]]

test_y_array=test_y.to_numpy()
dev_y_array=dev_y.to_numpy()



for max_depth in [15,20,30]:

    regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                              max_depth=max_depth,
                                                              random_state=0))
    regr_multirf.fit(train_X, train_y)

    # Predict on new data
    test_y_pre = regr_multirf.predict(test_X)
    dev_y_pre = regr_multirf.predict(dev_X)

    start_time = datetime.now()
    print("test")
    print_scores(test_y_array,test_y_pre)
    print("dev")
    print_scores(dev_y_array,dev_y_pre)
    end_time = datetime.now()
    print(end_time - start_time)
