from sklearn import linear_model
import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
train=pd.read_csv("/Users/saikrishna/Downloads/Solar/Task 1/predictors1_H.csv")
train.head()
train.shape
X_train=train.drop(['ZONEID','TIMESTAMP'], axis=1)
X_train.head()
X_train.shape
train2=pd.read_csv("/Users/saikrishna/Downloads/Solar/Task 1/train1.csv")
Y_train=train2.drop(['ZONEID','TIMESTAMP'],axis=1)
Y_train.head()
scoring = make_scorer(r2_score)
g_cv = GridSearchCV(DecisionTreeRegressor(random_state=0),
              param_grid={'min_samples_split': range(2, 10)},
              scoring=scoring, cv=5, refit=True)
g_cv.fit(X_train, Y_train)
g_cv.best_params_
test=pd.read_csv("/Users/saikrishna/Downloads/Solar/Task 1/April2013Inputs.csv")
test.head()
X_test=test.drop(['ZONEID','TIMESTAMP'], axis=1)
X_test.head()
test2=pd.read_csv("/Users/saikrishna/Downloads/Solar/Task 1/April2013outputs.csv")
test2.head()
Y_test=test2.drop(['ZONEID','Timestamp'],axis=1)
Y_test.head()
X_test.shape
Y_test.shape
mse = mean_squared_error(Y_test, g_cv.best_estimator_.predict(X_test))
rmse=math.sqrt(mse)
rmse
mean_absolute_error(Y_test, g_cv.best_estimator_.predict(X_test))
