''' 
Simple example for an Echo State Network in Python

Inspired by "A tutorial on training recurrent neural
networks, covering BPPT, RTRL, EKF and the
echo state network approach" by Herbert Jaeger
'''

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


transSteps = 100

trainSteps = 300 + transSteps
testSteps = 300 + transSteps

spectralRadius = 0.95

numOfInputs = 1
numOfHiddenUnits = 20
numOfOutputUnits = 1


input =  1/2 * np.sin(np.arange(trainSteps + 1)/4)


# initialize input, reservoir, and output matrices
W_inp_res = 0.2 * np.random.rand(numOfHiddenUnits, numOfInputs) - 0.1
W_res_res = 2.0 * np.random.rand(numOfHiddenUnits, numOfHiddenUnits) - 1.0
W_res_out = 2.0 * np.random.rand(numOfHiddenUnits, numOfOutputUnits) - 1.0 


# scaling the spectral radius of the reservoir
W_res_res = W_res_res / abs(linalg.eigvals(W_res_res)).max() * spectralRadius

# intialize internal states with zeros
X = np.zeros((numOfHiddenUnits, trainSteps))

# run the network on the input sequence
for t in xrange(trainSteps):
	X[:,t] = np.tanh(np.dot(W_res_res, X[:,t-1]).T + np.dot(W_inp_res, input[t]).T)
	
# (pseudo-)invert matrix with network activations, discard transients
X_inv = linalg.pinv(X[:,transSteps:trainSteps])

# calculate output weights for one step prediction
W_res_out = np.dot(input[transSteps + 1:trainSteps + 1], X_inv).T

# test on the same dataset for now (not very useful, but as a proof-of-concept)
input =  1/2 * np.sin(np.arange(trainSteps + 5)/4)

prediction = np.zeros(testSteps)

# testing
for t in xrange(testSteps):
	X[:,t] = np.tanh(np.dot(W_res_res, X[:,t-1]).T + np.dot(W_inp_res, input[t]).T)
	prediction[t] = np.dot(W_res_out, X[:,t])
	
# calculate error between one step prediction and desired input values	
mse = np.mean((input[transSteps + 1:testSteps + 1] - prediction[transSteps:testSteps]) ** 2)

print('MSE:', mse)

ax = np.arange(testSteps - transSteps)

# plot everything
plt.plot(ax,input[transSteps + 1:testSteps + 1])
plt.plot(ax,prediction[transSteps:testSteps])
plt.show()
