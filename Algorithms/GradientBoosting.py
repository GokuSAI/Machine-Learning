import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
train=pd.read_csv("/Users/apoorva/Downloads/Solar/Task 1/predictors1_H.csv")
train.head()
train.shape
X_train=train.drop(['ZONEID','TIMESTAMP'], axis=1)
X_train.head()
X_train.shape
train2=pd.read_csv("/Users/saikrishna/Downloads/Solar/Task 1/train1.csv")
Y_train=train2.drop(['ZONEID','TIMESTAMP'],axis=1)
Y_train.head(10)
Y_train.shape
params = {'n_estimators': 500, 'max_depth': 6,
        'learning_rate': 0.1, 'loss': 'huber','alpha':0.95}
clf = GradientBoostingRegressor(**params)
clf.fit(X_train, np.ravel(Y_train,order='A'))
test=pd.read_csv("/Users/saikrishna/Downloads/Solar/Task 1/April2013Inputs.csv")
test.head()
X_test=test.drop(['ZONEID','TIMESTAMP'], axis=1)
X_test.head()
X_test.shape
test2=pd.read_csv("/Users/saikrishna/Downloads/Solar/Task 1/April2013outputs.csv")
test2.head()
Y_train=test2.drop(['ZONEID','Timestamp'],axis=1)
Y_train.head()
Y_train.shape
mse = mean_squared_error(Y_train, clf.predict(X_test))
rmse=math.sqrt(mse)
rmse
mean_absolute_error(Y_train, clf.predict(X_test))
