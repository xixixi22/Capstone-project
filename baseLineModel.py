import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

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


def print_scores(y_array,pred_y):
    pearson_mean,_ = correlation(y_array, pred_y, 'pearson')
    spearman_mean,_ = correlation(y_array, pred_y, 'spearman')
    print(pearson_mean,spearman_mean)
    print(rmse(y_array, pred_y))
    

def knn(x,test):
    neigh = KNeighborsRegressor(n_neighbors=x)
    regr_multiknn= MultiOutputRegressor(neigh).fit(train_X,train_y)
    multiknn_pre = regr_multiknn.predict(test)
    print_score(test,multiknn_pre)
    
def randomSearch(base_model,random_grid):
    random = RandomizedSearchCV(MultiOutputRegressor(base_model),param_distributions = random_grid,
                               n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    random.fit(train_X, train_y)
    print(random.best_params_)
    best_random = random.best_estimator_
    pred_y_train = best_random.predict(train_X)
    print_scores(train_y_array,pred_y_train)
    pred_y_test = best_random.predict(test_X)
    print_scores(test_y_array,pred_y_test)
    pred_y_dev = best_random.predict(dev_X)
    print_scores(dev_y_array,pred_y_dev)
    
train = pd.DataFrame(pd.read_csv("data/train.csv"))
test = pd.DataFrame(pd.read_csv("data/test.csv"))
dev = pd.DataFrame(pd.read_csv("data/dev.csv"))

train_X,train_y=train[train.columns[978:]],train[train.columns[:978]]
test_X,test_y=test[test.columns[978:]],test[test.columns[:978]]
dev_X,dev_y=dev[dev.columns[978:]],dev[dev.columns[:978]]

test_y_array=test_y.to_numpy()
dev_y_array=dev_y.to_numpy()
print(train_X)



#random forest with random search
print("++++++++random search RF+++++++")
regr_RF = RandomForestRegressor()
n_estimators = [10,50,100,150]
max_depth = [10,15,20]
max_leaf_nodes = [3,5,7,9]
max_features=["auto", "sqrt", "log2"]
random_grid_RF = {
    'estimator__n_estimators':n_estimators,
    'estimator__max_depth':max_depth,
    'estimator__max_leaf_nodes':max_leaf_nodes,
    'estimator__max_features':max_features
}
randomSearch(regr_RF,random_grid_RF)

#elastic net with random search
regr_EN = ElasticNet()
alpha = [x for x in np.linspace(0.01,1,5)]
l1_ratio = [x for x in np.linspace(0.01,1,5)]

random_grid = {
    'estimator__alpha':alpha,
    'estimator__l1_ratio':l1_ratio
}
randomSearch(regr_EN,random_grid)

#knn
for i in range(1,19):
    knn(i,test_X)
    
#lassoCV 
alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
regr_multila = MultiOutputRegressor(LassoCV(cv = 5,alphas=alphas))
regr_multila.fit(train_X,train_y)
regr_multila.get_params
y_pred = regr_multila.predict(test_X)
print_scores(test_y_array,y_pred)

#ridgeCV
regr_multiri = MultiOutputRegressor(RidgeCV(cv = 5,alphas = alphas))
regr_multiri.fit(train_X,train_y)
regr_multiri.get_params
y_pred = regr_multiri.predict(test_X)
print_scores(test_y_array,y_pred)

#ridge
for i in alphas:
    regr_multir = MultipleOutputRegressor(Ridge(alpha = i))
    regr_multir.fit(train_X,train_y)
    y_pred1 = regr_multir.predict(test_X)
    print_scores(test_y_array,y_pred1)
    
# lasso
for i in alphas:
    regr_multir = MultipleOutputRegressor(lasso(alpha = i))
    regr_multir.fit(train_X,train_y)
    y_pred = regr_multir.predict(test_X)
    print_scores(test_y_array,y_pred)
