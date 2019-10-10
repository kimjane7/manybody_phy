import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from definitions import preprocess, exponential
from scipy.optimize import curve_fit
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():

    # three layered RNN
    step = 2
    num_units = 125
    activation_func = "relu"
    rnn = Sequential()
    rnn.add(SimpleRNN(units=num_units, input_shape=(step,1), activation=activation_func, return_sequences=True))
    rnn.add(SimpleRNN(units=num_units, input_shape=(step,num_units),  activation=activation_func, return_sequences=True))
    rnn.add(SimpleRNN(units=num_units, input_shape=(step,num_units),  activation=activation_func))
    rnn.add(Dense(1))
    rnn.summary()
    rnn.compile(loss='mse', optimizer='adam')
    
    g_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    color_list = ['indianred', 'sandybrown', 'orange', 'gold', 'yellowgreen', 'forestgreen', 'mediumturquoise', 'steelblue', 'royalblue', 'mediumpurple']
    
    # train over the low interaction flows
    for g in g_list[4:5]:
    
        datafile = "data/magnus_g"+str(g)+".dat"
        data = np.loadtxt(datafile, unpack=True)
        
        s = data[0]
        E = data[1]
        
        trainX, trainY = preprocess(E, step, 1)
        num_epochs = 200
        rnn.fit(trainX, trainY, epochs=num_epochs, batch_size=28, verbose=1)
    
    
    plt.figure(figsize=(10,8))
    fig = plt.figure(1)
    axes = plt.gca()
    axes.set_xlim([0.0,10.0])
    axes.tick_params(labelsize=12)
    
    # plot predictions
    for g in g_list:

        datafile = "data/magnus_g"+str(g)+".dat"
        data = np.loadtxt(datafile, unpack=True)
        
        s = data[0]
        E = data[1]
    
        testX, testY = preprocess(E, step, 1)
        prediction = rnn.predict(testX)
        
        plt.plot(s, E, linestyle='solid', label="data, g = "+str(g), color=color_list[g_list.index(g)])
        plt.plot(s[step:], prediction, linestyle='dashed', label="prediction, g = "+str(g),color=color_list[g_list.index(g)])

    plt.grid(True, alpha=0.2)
    plt.legend(loc=1, ncol=1, fontsize=12)
    plt.xlabel("Flow parameter s", fontsize=12)
    plt.ylabel("First diagonal element", fontsize=12)
    plt.title("Flow prediction by RNN for various interaction strengths g", fontsize=12)
    
    figname = "flow_prediction_g0.5.pdf"
    plt.savefig(figname, format='pdf')
    os.system('open '+figname)
    plt.clf()
        

if __name__ == "__main__":
    main()
