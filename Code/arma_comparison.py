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
import statsmodels.api as sm
# this is the nsteps ahead predictor function
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample


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
    res = pickle.load(open(path + '/datasets/fc_net_mnist/config_%i.pkl' %i, 'rb'))
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
    train_size = int(len(curve)*0.67)
    test_size = len(curve) - train_size
    
    # training and test sets
    train, test = np.array(curve[0:train_size]), np.array(curve[train_size:len(curve)])
    
    # model (0 makes it stationary)
    # ToDo check why it's not possible to train and test in the classic way !
    arma_model = sm.tsa.ARMA(train, order=(0, 2))    
    
    fit_res = arma_model.fit(trend='nc')


    predictions_train = arma_model.predict(train)

    # old school error calculation
    error = np.mean((train-predictions_train)**2)
    print('Train MSE: %.8f' % error)
    # plot
    pyplot.plot(test, label = 'Trainingset')
    pyplot.plot(predictions_train, color='red', label = 'Predictions')
    pyplot.legend()
    pyplot.savefig(path +'/lstm_plots/arma_train_curve_%i_train.png' %i)
    
    arma_model = sm.tsa.ARMA(test, order=(0, 2))    
    
    fit_res = arma_model.fit(trend='nc')

    predictions = arma_model.predict(test)

    # old school error calculation
    error = np.mean((test-predictions)**2)
    print('Test MSE: %.8f' % error)
    # plot
    pyplot.plot(test, label = 'Testset')
    pyplot.plot(predictions, color='red', label = 'Predictions')
    pyplot.legend()
    pyplot.savefig(path +'/lstm_plots/arma_train_curve_%i_testset.png' %i)