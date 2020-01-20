import time
import numpy as np
from definitions import preprocess_timeseries
from deepRNN import deepRNN



def avg_sigma(predictor, num_trials, *args):
    """ Calculate the average and variance of a prediction
        using specified predictor function over number of trials.
        Store and return loss functions for all trials. """

    # set seed
    np.random.seed(int(time.time()))

    # calculate avg and standard deviation of E
    # and store loss during trainings
    s_pred, avg_E, loss, num_params = predictor(*args)
    avg_E2 = np.square(avg_E)
    loss_trials = [loss]
    
    for i in range(num_trials-1):
    
        s_pred, E_pred, loss, num_params = predictor(*args)
        
        avg_E += E_pred
        avg_E2  += np.square(E_pred)
        loss_trials.append(loss)
        
    avg_E /= num_trials
    avg_E2 /= num_trials
    sigma_E = np.sqrt(avg_E2-np.square(avg_E))
    
    return s_pred, avg_E, sigma_E, loss_trials, num_params



def predict_remainder_flow(layers, units, activation_func, \
                           data, frac_train, num_timesteps, \
                           num_epochs, use_early_stopping=False):
    """ Preprocess time series data, build RNN, and train on
        specified fraction of data. Make prediction for remainder
        of flow, store loss, and return number of parameters. """

    # reshape data
    X, y = preprocess_timeseries(data[1], num_timesteps, 1)
    num_data = len(data[0])
    
    # split data
    num_train = int(frac_train*num_data)
    trainX = X[:num_train]
    trainy = y[:num_train]
    
    # train model
    RNN = deepRNN(layers, units, activation_func, num_timesteps)
    RNN.train(trainX, trainy, num_epochs, use_early_stopping)
    
    # predict rest of flow
    s_pred = data[0][num_timesteps:]
    E_pred = RNN.predict(trainX, num_data-num_timesteps).reshape(-1)
    
    # MSE loss during training
    loss = RNN.fit.history['loss']

    return s_pred, E_pred, loss, RNN.num_params
    

def predict_different_flow(layers, units, activation_func, \
                           data1, data2, frac_init, num_timesteps, \
                           num_epochs, use_early_stopping=False):
    """ Preprocess both data sets, build RNN, and train on entire data1.
        Make prediction for entire flow in data2, using a fraction of
        data2 to give initial starting points. """
    
    # reshape data
    X1, y1 = preprocess_timeseries(data1[1], num_timesteps, 1)
    X2, y2 = preprocess_timeseries(data2[1], num_timesteps, 1)
    num_data = len(data2[0])
    
    # split data2
    num_init = int(frac_init*num_data)
    X2_init = X2[:num_init]
    
    # train model on data1
    RNN = deepRNN(layers, units, activation_func, num_timesteps)
    RNN.train(X1, y1, num_epochs, use_early_stopping)
    
    # predict flow for data2
    s_pred = data2[0][num_timesteps:]
    E_pred = RNN.predict(X2_init, num_data-num_timesteps).reshape(-1)
    
    # MSE loss during training
    loss = RNN.fit.history['loss']
    
    return s_pred, E_pred, loss, RNN.num_params
    
                           
'''
def predict_entire_flow(layers, units, activation_func, \
                        num_timesteps, num_epochs,
                        train_params, test_params,)
'''
