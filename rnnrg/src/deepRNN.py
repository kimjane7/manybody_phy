import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Flatten
from keras.callbacks import EarlyStopping



class deepRNN:
    
    def __init__(self, layers, units, activation_func, num_timesteps):
        """ Build network with specified layer types and numbers of units.
            Activation function is the same for each layer.
            'num_timesteps' is the dimension of each data sample.
            Store total number of parameters in 'num_params' member. """

        # check compatibility of input parameters
        if len(layers) != len(units):
            raise ValueError('Give number of units for each layer.')
        
        self.num_timesteps = num_timesteps
        num_layers = len(layers)
        
        # build deep RNN
        self.model = Sequential()
        
        for i in range(num_layers):
        
            # input shape
            if i == 0:
                shape = (num_timesteps, 1)
            else:
                shape = (num_timesteps, units[i-1])
            
            # return sequences if not last recurrent layer
            return_seq = True
            if i == num_layers-1:
                return_seq = False
                
            # add layers
            if layers[i] == 'd':
                
                # flatten first if last layer is dense
                if i == num_layers-1:
                    self.model.add(Flatten(input_shape=shape))
                    self.model.add(Dense(units[i], activation=activation_func))
                    
                else:
                    self.model.add(Dense(units[i], activation=activation_func, \
                                         input_shape=shape))
            
            elif layers[i] == 's':
                self.model.add(SimpleRNN(units[i], activation=activation_func, \
                                         input_shape=shape, return_sequences=return_seq))
            
            elif layers[i] == 'l':
                self.model.add(LSTM(units[i], activation=activation_func, \
                                    input_shape=shape, return_sequences=return_seq))
            
            elif layers[i] == 'g':
                self.model.add(GRU(units[i], activation=activation_func, \
                                   input_shape=shape, return_sequences=return_seq))
            
            else:
                raise ValueError('Choose layers from list: Dense (d), \
                                  SimpleRNN (s), LSTM (l), GRU (g).')
                
        # single output
        self.model.add(Dense(1))
        
        # compile
        self.model.compile(loss='mse', optimizer='adam')
        
        #self.model.summary()
        self.num_params = self.model.count_params()
        

    
    def train(self, X, y, num_epochs, use_early_stopping):
        """ Train network on a data set X and targets y for a fixed
            number of epochs or using the EarlyStopping callback. """
    
        if use_early_stopping:
        
            ES = EarlyStopping(monitor='loss', mode='min', \
                               patience=30, verbose=0)
            
            self.fit = self.model.fit(X, y, epochs=num_epochs, \
                                      callbacks=[ES], verbose=0, \
                                      shuffle=False)
            
        else:
            self.fit = self.model.fit(X, y, epochs=num_epochs, \
                                      verbose=0, shuffle=False)
            
            
    
    def predict(self, X, total_iters):
        """ Use trained model to make prediction on time
            series data set X and extrapolate until total
            number of iterations is reached. """
    
        prediction = np.empty((total_iters,1))
        prediction[:X.shape[0]] = self.model.predict(X)
        
        for iter in range(X.shape[0], total_iters):
            input = np.empty((1, self.num_timesteps, 1))
            for i in range(self.num_timesteps):
                input[0, i] = prediction[iter-self.num_timesteps+i]
            prediction[iter] = self.model.predict(input)

        return prediction

