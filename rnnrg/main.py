import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from deepRNN import deepRNN
from predict import predict_flow

def main():


    layers = 'dssg'
    units = [100, 100, 100, 100]
    act_func = 'relu'
    step = 5
    g = 0.5
    frac_train = 0.1
    epochs = 100
    use_early_stopping = False
    
    rnn = deepRNN(layers, units, act_func, step)
    (s_train, prediction), (s, E) = predict_flow(rnn, g, frac_train, step, epochs, use_early_stopping)
    
    #plt.plot(rnn.fit.history['loss'])
    plt.plot(s, E, color='b', label='Data')
    plt.plot(s_train, prediction, color='r', label='Prediction')
    plt.axvspan(0.0, frac_train*s[-1], alpha=0.2, color='r', label='training region')
    plt.show()


if __name__ == "__main__":
    main()
