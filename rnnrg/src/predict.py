import time
import numpy as np
from definitions import preprocess_timeseries
from deepRNN import deepRNN


def get_magnus_data(g):
    """ Returns s and E from magnus data for given interaction parameter g"""

    datafile = "../data/magnus_ds0.01_g"+str(round(g,1))+".dat"
    data = np.loadtxt(datafile, unpack=True)

    return data[0], data[1]


def avg_sigma(predictor, num_trials, *args):
    """ Calculate the average and variance of a prediction
        using specified predictor function over number of trials.
        Store and return loss functions for all trials. """

    # set seed
    np.random.seed(int(time.time()))

    # calculate avg and standard deviation of E
    # and store loss during trainings
    s_pred, E_pred_avg, s, E, loss, num_params = predictor(*args)
    E2_pred_avg = np.square(E_pred_avg)
    loss_trials = [loss]
    
    for i in range(num_trials-1):
    
        s_pred, E_pred, s, E, loss, num_params = predictor(*args)
        
        E_pred_avg += E_pred
        E2_pred_avg  += np.square(E_pred)
        loss_trials.append(loss)
        
    E_pred_avg /= num_trials
    E2_pred_avg /= num_trials
    E_pred_sigma = np.sqrt(E2_pred_avg-np.square(E_pred_avg))
    
    return s_pred, E_pred_avg, E_pred_sigma, s, E, loss_trials, num_params


def predict_remainder_flow(layers, units, activation_func, \
                           g, frac_train, num_epochs, \
                           num_timesteps=2, num_pairing_params=None):
    """ Preprocess time series data, build RNN, and train on
        specified fraction of data. Make prediction for remainder
        of flow, store loss, and return number of parameters.
        Compatible with type 1 and type 2 networks. """
        
    # get data
    s, E = get_magnus_data(g)
    s_pred = s[num_timesteps:]

    # reshape data
    num_data = len(s)
    X, y = preprocess_timeseries(E, num_timesteps, 1)
    
    # split data
    num_train = int(frac_train*num_data)
    trainX = X[:num_train]
    trainy = y[:num_train]
    
    # train model and make prediction
    RNN = deepRNN(layers, units, activation_func, num_timesteps, num_pairing_params, num_outputs=1)
    
    if num_pairing_params:
        pairing_params = np.full((num_train, num_pairing_params), g)
        RNN.train([trainX, pairing_params], trainy, num_epochs)
        E_pred = RNN.predict([trainX, pairing_params], num_data-num_timesteps)
        
    else:
        RNN.train(trainX, trainy, num_epochs)
        E_pred = RNN.predict(trainX, num_data-num_timesteps)


    return s_pred, E_pred, s, E, RNN.fit.history['loss'], RNN.num_params
    
    

def predict_different_flow(layers, units, activation_func, \
                           g1, g2_list, frac_init, num_epochs, \
                           num_timesteps=2, num_pairing_params=None):
    """ Preprocess both data sets, build RNN, and train on data for g1.
        Make prediction of entire flow for all g's in g2_list, using a
        fraction of data to give initial points. """
        
    # get training data and preprocess
    s, E1 = get_magnus_data(g1)
    X1, y1 = preprocess_timeseries(E1, num_timesteps, 1)
    num_data = len(s)
    num_init = int(frac_init*num_data)
    
    # train model on g1 data
    RNN = deepRNN(layers, units, activation_func, num_timesteps)
    RNN.train(X1, y1, num_epochs)
    if num_pairing_params:
        RNN.train([X1, g], y1, num_epochs)
    else:
        RNN.train(X1, y1, num_epochs)
    
    # store data and prediction
    E = np.array((len(g2_list)+1, num_data))
    E_pred = np.array((len(g2_list)+1, num_data-num_timesteps))
    s_pred = s[num_timesteps:]
    E[0] = E1
    E_pred[0] = RNN.predict(X1[:num_init], num_data-num_timesteps)
    
    # predict flows for all g2 in g2_list
    for i in range(len(g2_list)):
        
        # get data and store
        s2, E2 = get_magnus_data(g2_list[i])
        E[i+1] = E2
        
        # reshape and split data
        X2, y2 = preprocess_timeseries(E2, num_timesteps, 1)
        X2_init = X2[:num_init]
    
        # predict flow
        E_pred[i+1] = RNN.predict(X2_init, num_data-num_timesteps)
        
    
    return s_pred, E_pred, s, E, RNN.fit.history['loss'], RNN.num_params
    
                        
'''
def predict_entire_flow(layers, units, activation_func, \
                        data_list, num_epochs, \
                        train_params_list, test_params_list):
    """ Build RNN and only train with pairing model parameters as inputs
        and entire flows as targets. Make predictions for entire flows
        using different pairing model parameters. """
    
    # reshape data
    num_pairing_params = len(train_params_list[0])
    num_data = len(data_list[0])
    num_train = len(data_list)
    trainX = np.zeros((num_train, num_pairing_params))
    trainY = np.zeros((num_train, num_data))
    
    for i in range(num_train):
    
        trainX[i] = train_params_list[i]
        trainY[i] = data_list[i][1]
    
    # train model on some pairing model parameters
    RNN = deepRNN(layers, units, activation_func, num_timesteps=None, \
                  num_pairing_params=num_pairing_params, num_outputs=num_data)
    RNN.train(trainX, trainY, num_epochs)

    # predict entire flows for different pairing model parameters
    num_test = len(test_params_list)
    testX = np.zeros((num_test, num_pairing_params))
    testY = np.zeros((num_test, num_data))
    
    for i in range(num_test):
        
        testX[i] = test_params_list[i]
        testY[i] = RNN.model.predict(testX).reshape(-1)
    
    # MSE loss during training
    loss = RNN.fit.history['loss']
    
    return data_list[0][0], testY, loss, RNN.num_params
'''
