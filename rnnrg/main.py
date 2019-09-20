import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM


##############################################################
# reshape linear sequence data into inputs and outputs for rnn
##############################################################

def preprocess(data, step):
    
    N_samples = len(data)-step
    X, Y = np.zeros((N_samples,step)), np.zeros(N_samples)
    for i in range(N_samples):
        j = i+step
        X[i,:] = data[i:j]
        Y[i] = data[j]

    X = np.reshape(X, (N_samples, 1, step))

    return X, Y


##############################################################
# main program
##############################################################

def main():

    # get energy sequence
    filename = "data/imsrg_magnus_euler_0.001_flow.dat"
    data = np.loadtxt(filename)
    s = data[:,0]
    E = data[:,1]

    # split sequence into training and testing set
    fraction_train = 0.1
    step = 3
    N_total = E.shape[0]
    N_train = round(fraction_train*N_total)
    train, test = E[:N_train], E[N_train:N_total]

    # reshape for Keras
    trainX, trainY = preprocess(train, step)
    testX, testY = preprocess(test, step)

    # make deep rnn
    rnn = Sequential()
    rnn.add(SimpleRNN(units=1000, input_shape=(1,step), activation='relu', return_sequences=True))
    rnn.add(SimpleRNN(units=100, input_shape=(N_train-step,1), activation='relu'))
    rnn.add(Dense(10, activation='relu'))
    rnn.add(Dense(1))
    rnn.compile(loss='mse', optimizer='adam')

    # fit to data
    rnn.fit(trainX, trainY, epochs=5000, batch_size=28, verbose=1)
    train_predict_rnn = rnn.predict(trainX)
    test_predict_rnn = rnn.predict(testX)
    prediction_rnn = np.concatenate((train_predict_rnn,test_predict_rnn), axis=0)
    
    # make deep LSTM with same structure
    lstm= Sequential()
    lstm.add(LSTM(units=1000, input_shape=(1,step), activation='relu', return_sequences=True))
    lstm.add(LSTM(units=100, input_shape=(N_train-step,1), activation='relu'))
    lstm.add(Dense(10, activation='relu'))
    lstm.add(Dense(1))
    lstm.compile(loss='mse', optimizer='adam')
    
    # fit to data
    lstm.fit(trainX, trainY, epochs=5000, batch_size=28, verbose=1)
    train_predict_lstm = lstm.predict(trainX)
    test_predict_lstm = lstm.predict(testX)
    prediction_lstm = np.concatenate((train_predict_lstm,test_predict_lstm), axis=0)

    # plot rnn and lstm predictions
    plt.plot(s, E, linewidth=5, label="data", color='blue')
    plt.plot(s[-len(prediction_rnn):], prediction_rnn, linewidth=5, label="deep RNN prediction", linestyle=':', color='red')
    plt.plot(s[-len(prediction_lstm):], prediction_lstm, linewidth=5, label="deep LSTM prediction", linestyle=':', color='green')
    plt.axvspan(0.0, s[N_train], color='orange', alpha=0.2, label="training region")
    plt.legend()
    plt.ylabel("Ground state energy E")
    plt.xlabel("flow parameter s")
    plt.xlim(0,10)
    plt.show()

if __name__ == "__main__":
    main()
