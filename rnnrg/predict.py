import numpy as np
from definitions import preprocess
from deepRNN import deepSimpleRNN, deepLSTM

def predict_flow(g, frac_train, RNN_type, units, activation_func, step):
    
    # get data
    datafile = "data/magnus_ds0.01_g"+str(g)+".dat"
    data = np.loadtxt(datafile, unpack=True)
    s = data[0]
    E = data[1]
    
    # reshape data
    X, y = preprocess(E, step, 1)
    
    # split data
    num_train = int(frac_train*len(s))
    s_train = s[step:]
    trainX = X[:num_train]
    trainy = y[:num_train]
    
    # make model
    if RNN_type == 'simple':
        RNN = deepSimpleRNN(units, activation_func, step)
    if RNN_type == 'lstm':
        RNN = deepLSTM(units, activation_func, step)
    
    # train model
    RNN.train(trainX, trainy)

    # predict for the rest of the flow
    prediction = RNN.model.predict(X)
    
    return (s_train, prediction), (s, E)
