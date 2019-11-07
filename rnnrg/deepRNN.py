from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from keras.callbacks import EarlyStopping

class deepSimpleRNN:
    
    def __init__(self, num_layers, units_list, activation_func, step):
    
        # check that units are provided for each layer
        if num_layers != len(units_list):
            raise ValueError("Provide number of units for each simple RNN layer.")
        
        # build deep RNN
        self.model = Sequential()
        
        # first RNN layer has input shape determined by step
        if num_layers == 1:
            self.model.add(SimpleRNN(units=units_list[0], \
                                     activation=activation_func, \
                                     input_shape=(step,1)))
                                 
        if num_layers >= 2:
            self.model.add(SimpleRNN(units=units_list[0], \
                                     activation=activation_func, \
                                     return_sequences=True, \
                                     input_shape=(step,1)))
                                     
        if num_layers >= 3:
            # interior RNN layers
            for n in range(1, num_layers-1):
                self.model.add(SimpleRNN(units=units_list[n], \
                                         activation=activation_func, \
                                         return_sequences=True, \
                                         input_shape=(step, units_list[n-1])))
        if num_layers >= 2:
            # last RNN layer
            self.model.add(SimpleRNN(units=units_list[-1], \
                                     activation=activation_func, \
                                     input_shape=(step, units_list[-2])))
        
        # single output
        self.model.add(Dense(1))
        
        # compile
        self.model.compile(loss='mse', optimizer='adam')
    
    def train(self, X, y):
        
        # automatic stopping of training
        ES = EarlyStopping(monitor='loss', mode='min', \
                           patience=20, verbose=0)
        
        # fit
        max_epochs = 500
        self.fit = self.model.fit(X, y, \
                                  epochs=max_epochs, \
                                  callbacks=[ES], \
                                  verbose=1)
        
 

class deepLSTM:
 
    def __init__(self, num_layers, units_list, activation_func, step):
    
        # check that units are provided for each layer
        if num_layers != len(units_list):
            raise ValueError("Provide number of units for each simple RNN layer.")
        
        # build deep RNN
        self.model = Sequential()
        
        # first RNN layer has input shape determined by step
        if num_layers == 1:
            self.model.add(LSTM(units=units_list[0], \
                                activation=activation_func, \
                                input_shape=(step,1)))
                                 
        if num_layers >= 2:
            self.model.add(LSTM(units=units_list[0], \
                                activation=activation_func, \
                                return_sequences=True, \
                                input_shape=(step,1)))
                                     
        if num_layers >= 3:
            # interior RNN layers
            for n in range(1, num_layers-1):
                self.model.add(LSTM(units=units_list[n], \
                                    activation=activation_func, \
                                    return_sequences=True, \
                                    input_shape=(step, units_list[n-1])))
        if num_layers >= 2:
            # last RNN layer
            self.model.add(LSTM(units=units_list[-1], \
                                activation=activation_func, \
                                input_shape=(step, units_list[-2])))
        
        # single output
        self.model.add(Dense(1))
        
        # compile
        self.model.compile(loss='mse', optimizer='adam')
    
    def train(self, X, y):
        
        # automatic stopping of training
        ES = EarlyStopping(monitor='loss', mode='min', \
                           patience=20, verbose=0)
        
        # fit
        max_epochs = 500
        self.fit = self.model.fit(X, y, \
                                  epochs=max_epochs, \
                                  callbacks=[ES], \
                                  verbose=1)
