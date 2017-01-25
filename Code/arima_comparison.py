# -*- coding: utf-8 -*-
"""
Comparing ARIMA with keras' RNN used for lc prediction
(AutoRegressive Integrated Moving Average)
"""

import numpy as np
from scipy import stats
import pickle
import os
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

'''
Loading data
'''

path = os.path.dirname(os.path.realpath(__file__))

N = 5
n_epochs = 100
configs = []
learning_curves = []
cost = []
i = 0
while len(configs) < N:
    res = pickle.load(open(path + '/learning_curve_prediction/datasets/fc_net_mnist/config_%i.pkl' %i, 'rb'))
    i += 1
    learning_curves.append(res['learning_curve'])
    configs.append(res['config'].get_array())
    cost.append(res['cost'])
    
configs = np.array(configs)
learning_curves = np.array(learning_curves)
cost = np.array(cost)

'''
Fitting and Predicting
'''

for i, curve in enumerate(learning_curves):
    # normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    curve = scaler.fit_transform(curve)
    train_size = int(len(curve)*0.67)
    test_size = len(curve) - train_size
    
    # training and test sets
    train, test = np.array(curve[0:train_size]), np.array(curve[train_size:len(curve)])
    
    '''
    fit an ARIMA(5,1,0) model.
    Params:
    lag value to 5 for autoregression, 
    difference order of 1 to make the time series stationary, 
    moving average model of 0.
    '''
    
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
        error = mean_squared_error(test, predictions)
        print('Test MSE: %.3f' % error)
        # plot
        pyplot.plot(test)
        pyplot.plot(predictions, color='red')
        pyplot.show()
    