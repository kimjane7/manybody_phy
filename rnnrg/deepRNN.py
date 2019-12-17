import numpy as np
np.random.seed(1337) # for reproducability
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
from keras.callbacks import EarlyStopping

class deepRNN:
    
    def __init__(self, layers, units, activation_func, step):

        if len(layers) != len(units):
            raise ValueError('Give number of units for each layer.')
        
        self.step = step
        num_layers = len(layers)
        
        # build deep RNN
        self.model = Sequential()
        
        for i in range(num_layers):
        
            # input shape
            if i == 0:
                shape = (step,1)
            else:
                shape = (step, units[i-1])
            
            # return sequences if not last layer
            return_seq = True
            if i == num_layers-1:
                return_seq = False
                
            # add layers
            if layers[i] == 'd':
                self.model.add(Dense(units[i], activation=activation_func, \
                                     input_shape=shape))
            
            if layers[i] == 's':
                self.model.add(SimpleRNN(units[i], activation=activation_func, \
                                         input_shape=shape, return_sequences=return_seq))
            
            if layers[i] == 'l':
                self.model.add(LSTM(units[i], activation=activation_func, \
                                    input_shape=shape, return_sequences=return_seq))
            
            if layers[i] == 'g':
                self.model.add(GRU(units[i], activation=activation_func, \
                                   input_shape=shape, return_sequences=return_seq))
       
        # single output
        self.model.add(Dense(1))
        
        # compile
        self.model.compile(loss='mse', optimizer='adam')
        
        self.model.summary()
        

    
    def train(self, X, y, epochs, use_early_stopping):
        
        if use_early_stopping:
        
            ES = EarlyStopping(monitor='loss', mode='min', \
                               patience=100, verbose=0)
            
            self.fit = self.model.fit(X, y, \
                                      epochs=epochs, \
                                      callbacks=[ES], \
                                      verbose=1, \
                                      shuffle=False)
            
        else:
            self.fit = self.model.fit(X, y, \
                                      epochs=epochs, \
                                      verbose=1, \
                                      shuffle=False)
            
    
    def predict(self, X, total_iters):
    
        prediction = np.empty((total_iters,1))
        prediction[:X.shape[0]] = self.model.predict(X)
        
        for iter in range(X.shape[0], total_iters):
            input = np.empty((1, self.step, 1))
            for i in range(self.step):
                input[0, i] = prediction[iter-self.step+i]
            prediction[iter] = self.model.predict(input)

        return prediction

