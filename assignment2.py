#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:17:58 2019

@author: salihemredevrim
"""

import pandas as pd
import matplotlib.pyplot as plt
from mat4py import loadmat
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense
from keras.layers import Dropout
import numpy
numpy.random.seed(1905)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid

#%%

#Assignment-2
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
               
    return data2

#%%
#Neural Network    
#Initially, single fully connected hidden layer is tried with selected activation function    
#Since it's a regression problem we don't need any transformation in the output layer
#Adam is selected as optimization method: 
#Others can be tried after checking performance of Adam, otherwise number of runs would be much more higher in grid search below)
#Dropout method is selected for regularization (due to the same reason above)
    
def simple_nn(num_nodes, k, activation_func, dropout_rate, init):
 
    #num_nodes: number of nodes in the hidden layer
    #k: number of variables/inputs coming from previous values (from 1 to 50)
    #activation_func: type of activation functions in the hidden layer (relu, sigmoid)
    #dropout_rate: drop out rate (warning: this is probability of turning off)
    #init: weight initialization (uniform, normal)
    
    #Create model
    model = Sequential()
    model.add(Dense(num_nodes, input_dim=k, kernel_initializer=init, activation=activation_func))  
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer=init, activation='linear'))
    
    #Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return model

#%%
#Different values for k will be tried to find the best window size (variables) from 1 to 50 
#Therefore, 200 observations (~%20) after time=50 were selected randomly, in order to validate all models by the same set
#CV in grid search takes much more time and we would like to evaluate best h-parameters and k values on the same data points  
#Since real test set will be provided later, we'll use this to choose the best h-parameters set
#Remaining observations will be used in training of NNs if they are not including empty variables (previous obs)

val_times = time_series[50:].sample(n=200, random_state=1905).sort_values('time')     

plt.plot(val_times.time, val_times.target)
plt.show()    

#%%
def prepare_datasets(data1, k): 
    
    #data1: time series data set 
    #k: number of variables/inputs 
    
    data2 = create_variables(data1, k)
    
    #take the rows in val_times as val_data and others as train_data
    val_data = data2[data2['time'].isin(val_times.time)]
    train_data = data2[~data2['time'].isin(val_times.time)]
    
    #create Xs and Ys
    y_train = pd.DataFrame(train_data.target)
    X_train = pd.DataFrame(train_data.drop(['time','target'], axis=1))
    
    #for training, eliminate observations with null variables located at the first k rows (they won't be used in training since they have null values)       
    X_train = X_train[k:].reset_index(drop=True) 
    y_train = y_train[k:].reset_index(drop=True)

    y_val = pd.DataFrame(val_data.target).reset_index(drop=True)
    X_val = pd.DataFrame(val_data.drop(['time','target'], axis=1)).reset_index(drop=True)
    
    return X_train, y_train, X_val, y_val
    
#%%
##Runs with grid search and evaluation on the validation set

def lets_go_grid(data1, epochs1, batch_size1):
    
    #initialize 
    output = pd.DataFrame(); 
    counter1 = 0
    k_list = [1, 5, 10, 15, 20, 30, 50]
    
    for k1 in k_list:
        print('go for: '+str(k1))
    
        #create train/test
        X_train, y_train, X_val, y_val = prepare_datasets(data1, k1)
    
        #define the grid search parameters and model 
        #1260 sets
        grid = ParameterGrid({"num_nodes": [3, 5, 10, 15, 20, 25, 30],
                          "k": [k1],
                          "activation_func":['relu'], #sigmoid was worse in initial trials so ignored
                          "dropout_rate": [0, 0.25],
                          "init": ['normal', 'uniform']})

        for params in grid:
            counter1 = counter1 + 1
            print(params)
            model = KerasRegressor(build_fn=simple_nn, epochs=epochs1, batch_size=batch_size1, verbose=0, **params)
            model.fit(X_train, y_train)
            
            #prediction on validation
            predictions = model.predict(X_val)
            predictions = pd.DataFrame(predictions)
            #evaluation
            comparison = pd.concat([y_val, predictions], axis = 1)
            comparison = comparison.rename(columns={0: 'predictions'}).reset_index(drop=False)

            comparison.plot(x='index', y=['target', 'predictions'], label=['target', 'predictions'])
            plt.show()

            mse = mean_squared_error(comparison.target,comparison.predictions)
            r_squared = r2_score(comparison.target,comparison.predictions)
            
            output1 = {'run': counter1, 
                       'k': k1, 
                       'num_nodes': params['num_nodes'],
                       'activation_func': params['activation_func'],
                       'dropout_rate': params['dropout_rate'],
                       'init': params['init'],
                       
                       'mse' : mse, 
                       'r2': r_squared 
                        }
            
            output1 = pd.DataFrame(output1, index=[0])
            
            output = pd.concat([output, output1], axis=0)
            
    return output

#%%
#Run grid search for 1-hidden layer NN (warning: it takes time, faster in Google colab)
grid_results1 = lets_go_grid(time_series, 150, 50)   

#to excel 
writer = pd.ExcelWriter('grid_results.xlsx', engine='xlsxwriter');
grid_results1.to_excel(writer, sheet_name= 'grid_results');
writer.save(); 

#%%
#Grid search evaluation of 1-hidden layer NN based on MSE

eval_results = pd.read_excel('grid_results.xlsx').reset_index(drop=True)

best_of_ks = eval_results.groupby(['k'])['mse'].min().reset_index(drop=False).sort_values('k')

plt.plot(best_of_ks.k, best_of_ks.mse)
plt.show()   

#top 10 h-parameters sets
top10 = eval_results.sort_values('mse')
top10.head(10)

#%%
del eval_results, best_of_ks, top10
#%%
#Best h-parameters for 1-hidden layer NN 

#num_nodes: 25
#k: 20
#activation_func: Relu
#dropout_rate: 0
#init: weight initialization; Normal

model = simple_nn(25, 20, 'relu', 0, 'normal')

#create train/validation
X_train, y_train, X_val, y_val = prepare_datasets(time_series, 20)

#fit on train set
model.fit(X_train, y_train, epochs=200, batch_size=50, verbose=1)

#validation set results 
predictions = model.predict(X_val)
predictions = pd.DataFrame(predictions)

comparison = pd.concat([y_val, predictions], axis = 1)
comparison = comparison.rename(columns={0: 'predictions'}).reset_index(drop=False)
comparison.plot(x='index', y=['target', 'predictions'], label=['target', 'predictions'])
plt.show()

mse_1hl = mean_squared_error(comparison.target,comparison.predictions)
r_squared_1hl = r2_score(comparison.target,comparison.predictions)

#MSE on validation set of best model of 1 hidden layer ~ 98
#R2 on validation set of best model of 1 hidden layer = 0.958
#%%

#Prediction of next 200 points recursively 

def predict_next_x(data1, k, x, model1):
    
    #data1: initial training data
    #k: number of previous points as variables
    #x: number of following points that we are predicting
    #model1: final selected model 
    
    #take last k points of training data 
    vertical_data = data1.tail(k).reset_index(drop=True)
    
    #initialize and append prediction points 
    empty = pd.DataFrame(index=range(x),columns=range(1)).reset_index(drop=False)
    empty = empty.rename(index=str, columns={"index": "time", 0: "target"})
    #initially target is 0
    empty['target'] = 0 
    
    vertical_data = vertical_data.append(empty).reset_index(drop=True)
    
   # horizontal_data = create_variables(vertical_data, k)
    
    for x1 in range(x):
        
        horizontal_data = create_variables(vertical_data, k)
        #take only variables for x1 
        variables = pd.DataFrame(horizontal_data[horizontal_data.time == x1])
        variables = variables.drop(['time','target'], axis=1)
        #predict test time of x1 
        prediction =  model1.predict(variables)
        #update target in x1 and recursion
        vertical_data.loc[vertical_data['time'] == x1, 'target'] = prediction
        
    return vertical_data, horizontal_data         

#%%        

vertical_data, horizontal_data = predict_next_x(time_series, 20, 200, model);        
        
last_predictions = vertical_data.tail(200).reset_index(drop=True)        
last_predictions = last_predictions.rename(columns={'target': 'predictions'}) 
        
#%%
#SECRET TEST SET DELETE THIS LATER
secret_test = pd.read_excel('secret_test.xlsx').reset_index(drop=False) 
secret_test = secret_test.rename(columns={"index": "time"})
    
plt.plot(secret_test.time, secret_test.target)
plt.show()

#%%
#Test on recursively predictions 

comparison_last = pd.concat([secret_test.target, last_predictions.predictions], axis = 1).reset_index(drop=False)
comparison_last.plot(x='index', y=['target', 'predictions'], label=['target', 'predictions'])
plt.show()

mse_1hl = mean_squared_error(comparison_last.target, comparison_last.predictions)
r_squared_1hl = r2_score(comparison_last.target, comparison_last.predictions)




