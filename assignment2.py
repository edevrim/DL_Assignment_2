#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:17:58 2019

@author: salihemredevrim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense
import numpy
from keras.regularizers import l2
numpy.random.seed(1905)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#%%

#Assignment-2:

#take dataset

ts = loadmat('Xtrain.mat')
         
ts2 = pd.DataFrame(ts)       

ts3 = pd.DataFrame(ts2.Xtrain.tolist(), columns=['target']).reset_index(drop=False)  
time_series = ts3.rename(index=str, columns={"index": "time"})

plt.plot(time_series.time, time_series.target)
plt.show()

#%%
del ts, ts2, ts3

#%%
#Functions for models 
#k is the window size for previous observed values which will be used as variables

def create_variables(data1, k):
    
    data2 = data1.copy(); 
    
    for l in range(k):
        data2[l] = data1['target'].shift(l+1)
    
#eliminate observations with null variables located at the first k rows (they won't be used in training)       
    data3 = data2[k:]            
    return data3

#%%
#Neural Network    
#Initially, single fully connected hidden layer is tried with selected activation function    
#Since it's a regression problem we don't need any transformation in output layer
#In addition, variables were not normalized come from the same source (laser or something like that) with the same scale   
#Adam is selected as optimization method 
    
def simple_nn(num_nodes, k, activation_func, reg, init):
    #num_nodes: number of nodes in the hidden layer (3, 5, 7, 10)
    #k: number of variables/inputs (different values from 1 to 50)
    #activation_func: type of activation function in the hidden layer (relu, sigmoid)
    #reg: regularization parameter (0.1, 1, 10 for L2 norm is selected)
    #init: weight initialization (uniform, normal)
    
    #Create model
    model = Sequential()
    #hidden layer
    model.add(Dense(num_nodes, input_dim=k, kernel_initializer=init, activation=activation_func, kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))   
    #output layer
    model.add(Dense(1, kernel_initializer=init))
    
    #Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'r-square'])
    return model

#%%

#Different values for k will be tried to find the best window size from 1 to 50 
#Therefore, 200 observations (~%20) after time=50 were selected randomly, in order to test all models by the same test set
#Actually real test set will be provided later, we'll use this to choose the best model  
#Remaining observations will be used in training of NNs if they are not including empty variables (previous obs)

test_times = time_series[50:].sample(n=200, random_state=1905).sort_values('time')     

plt.plot(test_times.time, test_times.target)
plt.show()    

#%%

def prepare_datasets(data1, k): 
    
    #data1: dataset 
    #k: number of variables/inputs 
    
    #take the rows in test_times as test_data and others as train_data
    test_data = data1[data1['time'].isin(test_times.time)]
    train_data = data1[~data1['time'].isin(test_times.time)]
    
    #create Xs and Ys
    y_train = pd.DataFrame(train_data.target).reset_index(drop=True)
    X_train = pd.DataFrame(train_data.drop(['time','target'], axis=1)).reset_index(drop=True)

    y_test = pd.DataFrame(test_data.target).reset_index(drop=True)
    X_test = pd.DataFrame(test_data.drop(['time','target'], axis=1)).reset_index(drop=True)
    
    return X_train, y_train, X_test, y_test
    
#%%
#Runs with grid search and evaluation on the test set

def lets_go(data1, k, num_nodes, activation_func, reg, init, epochs1, batch_size1):
    #epochs1: number of iterations
    #batch_size1: size for gradient descent
    
    #create train/test
    X_train, y_train, X_test, y_test = prepare_datasets(data1, k)
    
    #Fit the model
    model = simple_nn(num_nodes, k, activation_func, reg, init)
    
    model.fit(X_train, y_train, epochs=epochs1, batch_size=batch_size1)
    
    #Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#%%

#Fit the model
model = simple_nn(5, 10, 'relu', 0.01)    
model.fit(X_train, y_train, epochs=50, batch_size=20)

#Calculate predictions on test set
predictions = model.predict(X_test)
predictions = pd.DataFrame(predictions)

comparison = pd.concat([y_test, predictions], axis = 1)
comparison = comparison.rename(columns={0: 'predictions'}).reset_index(drop=False)

comparison.plot(x='index', y=['target', 'predictions'], label=['target', 'predictions'])
plt.show()

mse = mean_squared_error(comparison.target,comparison.predictions)
r_squared = r2_score(comparison.target,comparison.predictions)


#grid search and find best models per k 
k = 50
model = KerasRegressor(build_fn=simple_nn, epochs=100, batch_size=20, verbose=0)

for k1 in range(k):
    print('go for: '+str(k1))
    
    #create train/test
    X_train, y_train, X_test, y_test = prepare_datasets(time_series, k1)
    
    # define the grid search parameters
    num_nodes = [3, 5, 7, 10]
    activation_func = ['relu', 'sigmoid']
    reg = ['0.01', '0.1', '1', '10']
    init = ['uniform', 'normal']
    
    #grid search
    #A default cross-validation of 3 was used in GS
    param_grid = dict(num_nodes=num_nodes, k=k, activation_func=activation_func, reg=reg, init=init)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)
    best_params = grid_result.best_params_
    
    
    
    
    
    
    


