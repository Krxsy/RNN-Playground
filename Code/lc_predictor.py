# -*- coding: utf-8 -*-
"""
Playing around with the learning curve dataset in combination with different
RNNs as predictors.  
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import theano
from keras.models import Sequential
from keras.initializations import lecun_uniform
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import RMSprop
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


'''
converting an array into x,y dataset matrix
'''
def create_data(data, look_back = 1):
    X_data, Y_data = [], []
    for i in range(len(data)- look_back-1):
        a = data[i:(i+look_back)]
        X_data.append(a)
        Y_data.append(data[i+look_back])
    return np.array(X_data), np.array(Y_data)
    
'''
RNN models
'''
# LSTM
def _LSTM(trainX, trainY, testX, testY):
    model = Sequential()
    model.add(LSTM(4, input_shape=trainX.shape[1:]))
    model.add(Dense(1, activation='tanh'))

    model.compile(optimizer = 'adam', 
                  loss='mean_squared_error',
                  metrics=["accuracy"])
    
    print('Train...')
    model.fit(trainX, trainY,
              batch_size=1,
              nb_epoch=50,
              validation_data=[testX, testY],
              show_accuracy=False)

    print('Predicting...')
    train_preds = model.predict(trainX)
    test_preds = model.predict(testX)
    
    print(train_preds)
    # calculate error
    trainScore = mean_squared_error(trainY, train_preds)
    print('Train Score: %.2f MSE' % (trainScore))
    testScore = mean_squared_error(testY, test_preds)
    print('Test Score: %.2f MSE' % (testScore))
    
    return train_preds, test_preds
    
''''
Loading Data and Testing implementations
'''
path = os.path.dirname(os.path.realpath(__file__))

N = 4000
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

for i, curve in enumerate(learning_curves):
    # splitting into training and testing set & normalize with MinMax scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    curve = scaler.fit_transform(curve)
    train_size = int(len(curve)*0.67)
    test_size = len(curve) - train_size
    
    train, test = np.array(curve[0:train_size]), np.array(curve[train_size:len(curve)])
    
    # modelling actual dataset
    look_back = 1 # number of previous time steps to use as input variables to predict the next time period
    X_train, Y_train = create_data(train, look_back)
    X_test, Y_test = create_data(test, look_back)
    
    # reshape input for models in 3D (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    print('X_train shape: ', X_train.shape)
    print('X_test shape: ', X_test.shape)

    train_preds, test_preds = _LSTM(X_train, Y_train, X_test, Y_test)

    # plotting:
    # invert predictons
    train_preds = scaler.inverse_transform(train_preds)
    Y_train = scaler.inverse_transform([Y_train])
    test_preds = scaler.inverse_transform(test_preds)
    Y_test = scaler.inverse_transform([Y_test])
    
    
    #shift train predictions for plotting
    train_preds = train_preds.flatten()
    test_preds = test_preds.flatten()
    train_preds_plot = np.empty_like(curve)
    train_preds_plot[:] = np.nan
    train_preds_plot[look_back:len(train_preds)+ look_back] = train_preds

    # shift test predictions for plotting
    test_preds_plot = np.empty_like(curve)
    test_preds_plot[:] = np.nan
    test_preds_plot[len(train_preds)+(look_back*2)+1 : len(curve)-1] = test_preds
    
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(curve))
    plt.plot(train_preds_plot)
    plt.plot(test_preds_plot)
    plt.savefig(path +'/plots/curve_%i.png' %i)
    

'''
plot_subset = 50
t_idx = np.arange(1, n_epochs+1)

[plt.plot(t_idx, lc) for lc in learning_curves[:plot_subset]]
lt.title('subset of learning curves from cnn data')
plt.xlabel('number of epochs')
plt.ylabel('validation error')
plt.show()
'''