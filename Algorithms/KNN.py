import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def load_data(n=1):
    path=os.getcwd()
    print(path)
    a=''
    if n==15:
        a=path+'/Solar'+'/Solar'+'/Solution to Task 15'+'/Solution to Task 15.csv'
    else:
        a=path+'/Solar'+'/Task '+str(n)+'/train'+str(n)+'.csv'
        b=path+'/Solar'+'/Task '+str(n)+'/predictors'+str(n)+'_H.csv'
    y_train=list(pd.read_csv(a)['POWER'].values)         #365*24*3
    X=pd.read_csv(b).iloc[:,2:].values      #365*24*3+test_inputs
    X_train=X[0:len(y_train)]
    return list(X_train),list(y_train)
    def calculate_rmse_mae(y_actual, y_predicted):
    print('RMSE\t\t\t'+'MAE')
    print(mean_squared_error(y_actual, y_predicted)**.5,mean_absolute_error(y_actual, y_predicted))
    def main():
    X_train=[]
    y_train=[]
    y_hat=[]
    y_actual=[]
    #Initial fitting/training
    X_train,y_train=load_data(1)#Train
    X_test,y_test=load_data(2)  #Test
    knr_regressor=KNeighborsRegressor(n_neighbors=5,metric='minkowski',weights=input('Enter uniform/distance model: '))
    knr_regressor.fit(X_train,y_train)
    #Predicting and simultaneously fitting
    batch_size=30
    i=0
    for i in range(len(X_test)//batch_size):
            if i % 30==0:
                total_steps=len(X_test)//batch_size
                print('Progress: ',"%.2f" % ((i/total_steps)*100),'%')
            X=X_test[i*batch_size:i*batch_size+batch_size]
            y=y_test[i*batch_size:i*batch_size+batch_size]
            y_hat.append(knr_regressor.predict(X))
            y_actual.append(y)
            X_train.extend(X)
            y_train.extend(y)
            knr_regressor.fit(X_train,y_train)
    #Results
    calculate_rmse_mae(y_actual,y_hat)
    #pinball(y_actual,y_hat)
    if __name__ == '__main__':
    main()
