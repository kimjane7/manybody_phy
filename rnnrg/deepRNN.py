import numpy as np
np.random.seed(1337) # for reproducability
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from keras.callbacks import EarlyStopping

class deepSimpleRNN:
    
    def __init__(self, units, activation_func, step):
    
        num_layers = len(units)
        
        # build deep RNN
        self.model = Sequential()
        
        # first layer
        if num_layers == 1:
            self.model.add(SimpleRNN(units=units[0], \
                                     activation=activation_func, \
                                     input_shape=(step,1)))
                                 
        if num_layers >= 2:
            self.model.add(SimpleRNN(units=units[0], \
                                     activation=activation_func, \
                                     return_sequences=True, \
                                     input_shape=(step,1)))
        # interior layers
        if num_layers >= 3:
            for n in range(1, num_layers-1):
                self.model.add(SimpleRNN(units=units[n], \
                                         activation=activation_func, \
                                         return_sequences=True, \
                                         input_shape=(step, units[n-1])))
        # last layer
        if num_layers >= 2:
            self.model.add(SimpleRNN(units=units[-1], \
                                     activation=activation_func, \
                                     input_shape=(step, units[-2])))
        
        # single output
        self.model.add(Dense(1))
        
        # compile
        self.model.compile(loss='mse', optimizer='adam')
    
    def train(self, X, y):
        
        # automatic stopping of training
        ES = EarlyStopping(monitor='loss', mode='min', \
                           patience=20, verbose=0)
        
        # fit
        max_epochs = 1000
        self.fit = self.model.fit(X, y, \
                                  epochs=max_epochs, \
                                  callbacks=[ES], \
                                  verbose=1, \
                                  shuffle=False)
        
 

class deepLSTM:
 
    def __init__(self, units, activation_func, step):
    
        num_layers = len(units)
        
        # build deep RNN
        self.model = Sequential()
        
        # first layer
        if num_layers == 1:
            self.model.add(LSTM(units=units[0], \
                                activation=activation_func, \
                                input_shape=(step,1)))
                                 
        if num_layers >= 2:
            self.model.add(LSTM(units=units[0], \
                                activation=activation_func, \
                                return_sequences=True, \
                                input_shape=(step,1)))
                           
        # interior layers
        if num_layers >= 3:
            for n in range(1, num_layers-1):
                self.model.add(LSTM(units=units[n], \
                                    activation=activation_func, \
                                    return_sequences=True, \
                                    input_shape=(step, units[n-1])))
        # last RNN layer
        if num_layers >= 2:
            # last RNN layer
            self.model.add(LSTM(units=units[-1], \
                                activation=activation_func, \
                                input_shape=(step, units[-2])))
        
        # single output
        self.model.add(Dense(1))
        
        # compile
        self.model.compile(loss='mse', optimizer='adam')
    
    def train(self, X, y):
        
        # automatic stopping of training
        ES = EarlyStopping(monitor='loss', mode='min', \
                           patience=20, verbose=0)
        
        # fit
        max_epochs = 1000
        self.fit = self.model.fit(X, y, \
                                  epochs=max_epochs, \
                                  callbacks=[ES], \
                                  verbose=1, \
                                  shuffle=False)
