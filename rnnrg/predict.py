import numpy as np
from definitions import preprocess
from deepRNN import deepRNN

def predict_flow(RNN, g, frac_train, step, epochs=200, use_early_stopping=False):
    
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
    
    # train model
    RNN.train(trainX, trainy, epochs, use_early_stopping)

    # predict for the rest of the flow
    prediction = RNN.predict(trainX, len(s)-step)
    
    return (s_train, prediction), (s, E)
