from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import math
train=pd.read_csv("/Users/apoorva/Downloads/Solar/Task 1/predictors1_H.csv")
train.head()
train.shape
X_train=train.drop(['ZONEID','TIMESTAMP'], axis=1)
X_train.head()
X_train.shape
train2=pd.read_csv("/Users/apoorva/Downloads/Solar/Task 1/train1.csv")
Y_train=train2.drop(['ZONEID','TIMESTAMP'],axis=1)
Y_train.head(10)
Y_train.shape
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],'randomforestregressor__max_depth': [None, 5, 3, 1]}
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(X_train, np.ravel(Y_train,order='A'))
clf.refit
test=pd.read_csv("/Users/apoorva/Downloads/Solar/Task 1/April2013Inputs.csv")
test.head()
X_test=test.drop(['ZONEID','TIMESTAMP'], axis=1)
X_test.head()
X_test.shape
test2=pd.read_csv("/Users/apoorva/Downloads/Solar/Task 1/April2013outputs.csv")
test2.head()
Y_train=test2.drop(['ZONEID','Timestamp'],axis=1)
Y_train.head()
Y_train.shape
mse = mean_squared_error(Y_train, clf.predict(X_test))
rmse=math.sqrt(mse)
rmse
mean_absolute_error(Y_train, clf.predict(X_test))
