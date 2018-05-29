
#Gradient Boosting

import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
train=pd.read_csv("/Users/apoorva/Downloads/Solar/group/predictors1_H.csv")
train.head()
train.shape
X_train=train.drop(['ZONEID','TIMESTAMP','Hour'], axis=1)
X_train.head()
X_train.shape
train2=pd.read_csv("/Users/apoorva/Downloads/Solar/group/train1.csv")
Y_train=train2.drop(['ZONEID','TIMESTAMP'],axis=1)
Y_train.head(10)
Y_train.shape
params = {'n_estimators': 500, 'max_depth': 6,
        'learning_rate': 0.1, 'loss': 'huber','alpha':0.95}
clf = GradientBoostingRegressor(**params)
clf.fit(X_train, np.ravel(Y_train,order='A'))
test=pd.read_csv("/Users/apoorva/Downloads/Solar/group/April2013Inputs.csv")
test.head()
X_test=test.drop(['ZONEID','TIMESTAMP','Hour'], axis=1)
X_test.head()
X_test.shape
test2=pd.read_csv("/Users/apoorva/Downloads/Solar/group/April2013outputs.csv")
test2.head()
Y_train_rf=test2.drop(['ZONEID','Timestamp'],axis=1)
Y_train_rf.head()
Y_train_rf.shape
rms = mean_squared_error(Y_train_rf, clf.predict(X_test))
mean_absolute_error(Y_train_rf, clf.predict(X_test))
