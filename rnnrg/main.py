import sys
import numpy as np
import matplotlib.pyplot as plt
from definitions import preprocess, exponential
from scipy.optimize import curve_fit
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM


def main():

    # command line inputs
    method = sys.argv[1]
    activation_func = sys.argv[2]
    s_train = float(sys.argv[3])
    step = int(sys.argv[4])

    # only train for s < 2.5 b/c data is evenly spaced
    s_train = min(s_train,2.5)
    
    # data and output files
    datafile = "data/"+method+"_euler_0.001_flow.dat"
    outfile = "results/"+method+"_"+activation_func+"_train"+str(round(s_train,2))+"_step"+str(step)+".dat"
    
    # extract data
    data = np.loadtxt(datafile)
    s = data[:,0]
    if "imsrg" in method:
        eigs = data[:,1]
        num_eigs = 1
    else:
        eigs = data[:,2:]
        num_eigs = eigs.shape[1]
    num_pts_total = eigs.shape[0]
    
    # split data into training and testing sets
    num_pts_train = [i for i in range(num_pts_total) if abs(s[i]-s_train) < 0.0025][0]
    train, test = eigs[:num_pts_train], eigs[num_pts_train:num_pts_total]
    num_samples = num_pts_train-step
    
    # reshape sets for Keras
    trainX, trainY = preprocess(train, step, num_eigs)
    testX, testY = preprocess(test, step, num_eigs)
        
    # make deep RNN
    num_units = 100
    num_epochs = 2000
    rnn = Sequential()
    rnn.add(SimpleRNN(units=num_units, input_shape=(step,num_eigs), activation=activation_func, return_sequences=True))
    rnn.add(SimpleRNN(units=num_units, input_shape=(step,num_units),  activation=activation_func))
    rnn.add(Dense(num_eigs))
    rnn.compile(loss='mse', optimizer='adam')
    
    # fit RNN to data
    rnn.fit(trainX, trainY, epochs=num_epochs, batch_size=20, verbose=1)
    train_predict_rnn = rnn.predict(trainX)
    test_predict_rnn = rnn.predict(testX)
    prediction_rnn = np.concatenate((train_predict_rnn,test_predict_rnn), axis=0)

    
    # make deep LSTM
    lstm = Sequential()
    lstm.add(SimpleRNN(units=num_units, input_shape=(step,num_eigs), activation=activation_func, return_sequences=True))
    lstm.add(SimpleRNN(units=num_units, input_shape=(step,num_units),  activation=activation_func))
    lstm.add(Dense(num_eigs))
    lstm.compile(loss='mse', optimizer='adam')
    
    # fit LSTM to data
    lstm.fit(trainX, trainY, epochs=num_epochs, batch_size=20, verbose=1)
    train_predict_lstm = lstm.predict(trainX)
    test_predict_lstm = lstm.predict(testX)
    prediction_lstm = np.concatenate((train_predict_lstm,test_predict_lstm), axis=0)
     
    # fit data to exponential for each eigenvalue and plot
    if num_eigs > 1:
    
        plt.plot(s, eigs[:,0], linewidth=2, label="data", color='orange', marker='o')
        plt.plot(s[step:-step], prediction_rnn[:,0],linewidth=1, label="RNN prediction", color='red')
        plt.plot(s[step:-step], prediction_lstm[:,0],linewidth=1, label="LSTM prediction", color='blue')
        
        for j in range(1,num_eigs):
            plt.plot(s, eigs[:,j], linewidth=2, color='orange', marker='o')
            plt.plot(s[step:-step], prediction_rnn[:,j],linewidth=1, color='red')
            plt.plot(s[step:-step], prediction_lstm[:,j],linewidth=1, color='blue')
    else:
        train_params, cov = curve_fit(exponential, s[:num_pts_train], eigs[:num_pts_train], p0=(1,2,1))
        train_exp_fit = exponential(s, *train_params)
        plt.plot(s, eigs, linewidth=5, label="data", color='steelblue')

    
    
    plt.axvspan(0.0, s[num_pts_train], color='red', alpha=0.2, label="training region")
    plt.legend(loc=1)
    plt.ylabel("Eigenvalues")
    plt.xlabel("Flow parameter s")
    plt.xlim(0,0.4)
    plt.show()
    
    print(eigs[-1]-prediction_rnn[-1])
    print(eigs[-1]-prediction_lstm[-1])

if __name__ == "__main__":
    main()
