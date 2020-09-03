import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
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
def precision_k(label_test, label_predict, k):
    num_pos = 100
    num_neg = 100
    label_test = np.argsort(label_test, axis=1)
    label_predict = np.argsort(label_predict, axis=1)
    precision_k_neg = []
    precision_k_pos = []
    neg_test_set = label_test[:, :num_neg]
    pos_test_set = label_test[:, -num_pos:]
    neg_predict_set = label_predict[:, :k]
    pos_predict_set = label_predict[:, -k:]
    for i in range(len(neg_test_set)):
        neg_test = set(neg_test_set[i])
        pos_test = set(pos_test_set[i])
        neg_predict = set(neg_predict_set[i])
        pos_predict = set(pos_predict_set[i])
        precision_k_neg.append(len(neg_test.intersection(neg_predict)) / k)
        precision_k_pos.append(len(pos_test.intersection(pos_predict)) / k)
    return np.mean(precision_k_neg), np.mean(precision_k_pos)


def rmse(label_test, label_predict):
    return np.sqrt(mean_squared_error(label_test, label_predict))

def print_scores(y_array,pred_y):
    pearson_mean,_ = correlation(y_array, pred_y, 'pearson')
    spearman_mean,_ = correlation(y_array, pred_y, 'spearman')
    print(pearson_mean,spearman_mean)
    print(rmse(y_array, pred_y))
    for k in precision_degree:
        precision_neg, precision_pos = precision_k(y_array, pred_y, k)
        print("Precision@%d Positive: %.4f" % (k, precision_pos))
        print("Precision@%d Negative: %.4f" % (k, precision_neg))

def knn(x):
    neigh = KNeighborsRegressor(n_neighbors=x)
    regr_multiknn= MultiOutputRegressor(neigh).fit(train_X,train_y)
    test_pre = regr_multiknn.predict(test_X)
    print_scores(test_y_array,test_pre)
    print("dev:============")
    dev_pre = regr_multiknn.predict(dev_X)
    print_scores(dev_y_array, dev_pre)

train = pd.DataFrame(pd.read_csv("data/train.csv"))
test = pd.DataFrame(pd.read_csv("data/test.csv"))
dev = pd.DataFrame(pd.read_csv("data/dev.csv"))

train_X,train_y=train[train.columns[978:]],train[train.columns[:978]]
test_X,test_y=test[test.columns[978:]],test[test.columns[:978]]
dev_X,dev_y=dev[dev.columns[978:]],dev[dev.columns[:978]]

test_y_array=test_y.to_numpy()
dev_y_array=dev_y.to_numpy()
print(train_X)
precision_degree = [10, 20, 50, 100]

# for i in [0.7,0.8,0.9]:
#     clf = linear_model.Lasso(alpha=i)
#     start_time = datetime.now()
#     multi_lasso = MultiOutputRegressor(clf).fit(train_X,train_y)
#     print("================",i)
#     print("test")
#     l_test=multi_lasso.predict(test_X)
#     print("dev")
#     l_dev = multi_lasso.predict(dev_X)
#
#     print_scores(test_y_array,l_test)
#     print_scores(dev_y_array,l_dev)
#     end_time = datetime.now()
#     print(end_time - start_time)
for i in range(10，20):
    clf = linear_model.Ridge(alpha=i)
    start_time = datetime.now()
    multi_ridge = MultiOutputRegressor(clf).fit(train_X, train_y)
    print("================", i)
    print("test")
    l_test = multi_ridge.predict(test_X)
    print("dev")
    l_dev = multi_ridge.predict(dev_X)
    print_scores(test_y_array, l_test)
    print_scores(dev_y_array, l_dev)
    end_time = datetime.now()
    print(end_time - start_time)
