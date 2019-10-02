import numpy as np

##############################################################
# reshape sequence data into inputs and outputs for rnn
##############################################################

def preprocess(data, step, number):
    
    N_samples = data.shape[0]-step
    X, Y = np.zeros((N_samples, step, number)), np.zeros((N_samples, number))
    
    for i in range(N_samples):
        j = i+step
        X[i,:] = data[i:j]
        Y[i] = data[j]

    return X, Y
    
##############################################################
# compute exponential for curve fitting
##############################################################

def exponential(x, a, b, c):

    return a*np.exp(-b*x)+c
