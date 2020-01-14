import numpy as np



def exponential(x, a, b, c):
    """ Generic exponential for curve fitting. """

    return a*np.exp(-b*x)+c



def preprocess_timeseries(data, num_timesteps, num_features):
    """ Reshape time series data into 3D array with format
        (num_samples, num_timesteps, num_features) for RNN. """
    
    num_samples = data.shape[0]-num_timesteps
    X = np.zeros((num_samples, num_timesteps, num_features))
    Y = np.zeros((num_samples, num_features))
    
    for i in range(num_samples):
        j = i+num_timesteps
        X[i,:] = np.reshape(data[i:j], (num_timesteps, num_features))
        Y[i] = data[j]
    
    return X, Y
