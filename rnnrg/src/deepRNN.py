import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Flatten, Input, Concatenate
from keras.callbacks import EarlyStopping

class deepRNN:
    
    def __init__(self, layers, units, activation_func, \
                 num_timesteps=None, num_pairing_params=None, num_outputs=1):

        # check compatibility of input parameters
        if len(layers) != len(units):
            raise ValueError('Give number of units for each layer.')
            
        # type1
        if num_timesteps and num_pairing_params==None and num_outputs==1:
            self.build_RNN_one_output(layers, units, activation_func, num_timesteps)
        
        # type2
        elif num_timesteps and num_pairing_params and num_outputs==1:
            self.build_branched_RNN(layers, units, activation_func, num_timesteps, num_pairing_params)
    
        # type3
        elif num_timesteps==None and num_pairing_params and num_outputs>1:
            self.build_RNN_vector_output(layers, units, activation_func, num_pairing_params, num_outputs)
        
        else:
            raise ValueError('Network type not available.')
        
        # compile
        self.model.compile(loss='mse', optimizer='adam')
        self.model.summary()
        self.num_params = self.model.count_params()
    
    
    
    def build_partial_RNN(self, layers, units, activation_func, input_len):
        """ builds and returns a partial network with specified structure """
     
        num_layers = len(layers)
        model = Sequential()
        
        for i in range(num_layers):
        
            # input shape
            if i == 0:
                shape = (input_len, 1)
            else:
                shape = (input_len, units[i-1])
            
            # return sequences if not last recurrent layer
            return_seq = True
            if i == num_layers-1:
                return_seq = False
                
            # add layers
            if layers[i] == 'd':
                
                # flatten first if last layer is dense
                if i == num_layers-1:
                    model.add(Flatten(input_shape=shape))
                    model.add(Dense(units[i], activation=activation_func))
                    
                else:
                    model.add(Dense(units[i], activation=activation_func, \
                                    input_shape=shape))
            
            elif layers[i] == 's':
                model.add(SimpleRNN(units[i], activation=activation_func, \
                                    input_shape=shape, return_sequences=return_seq))
            
            elif layers[i] == 'l':
                model.add(LSTM(units[i], activation=activation_func, \
                               input_shape=shape, return_sequences=return_seq))
            
            elif layers[i] == 'g':
                model.add(GRU(units[i], activation=activation_func, \
                              input_shape=shape, return_sequences=return_seq))
            
            else:
                raise ValueError('Choose layers from list: Dense (d), \
                                  SimpleRNN (s), LSTM (l), GRU (g).')
            
            return model
        
    
    def build_RNN_one_output(self, layers, units, activation_func, num_timesteps):
        """ type1:  - use preprocessed time series data as inputs.
                    - each sample in data has dimension 'num_timesteps'
                    - predict next number in time series for each sample """
    
        # build RNN with one output
        self.model = self.build_partial_RNN(layers, units, activation_func, num_timesteps)
        self.model.add(Dense(1, activation=activation_func))
        
    
    def build_branched_RNN(self, layers, units, activation_func, num_timesteps, num_pairing_params):
        """ type2:  - use preprocessed time series data and three pairing model
                      parameters as inputs to two separate branches
                    - merge branches after data passes through RNN layers
                    - predict next number in time series for each sample """
        
        # data branch
        data_branch = self.build_partial_RNN(layers, units, activation_func, num_timesteps)
        
        # parameter branch
        param_branch = Sequential()
        param_branch.add(Dense(num_pairing_params, activation=activation_func, \
                               input_shape=(num_pairing_params, 1)))
        
        # merge branches
        merged = Concatenate()([data_branch.output, param_branch.output])
        merged = Dense(1, activation=activation_func)(merged)
        
        # build final model
        self.model = Model([data_branch.input, param_branch.input], merged)
        
      
    def build_RNN_vector_output(self, layers, units, activation_func, num_pairing_params, num_outputs):
    
        # build RNN with vector output
        self.model = self.build_partial_RNN(layers, units, activation_func, num_pairing_params)
        self.model.add(Dense(num_outputs, activation=activation_func))
    
    
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



layers = 'dls'
units = [10, 10, 10]
activation_func = 'linear'


type1 = deepRNN(layers, units, activation_func, num_timesteps=2, num_pairing_params=None, num_outputs=1)
type2 = deepRNN(layers, units, activation_func, num_timesteps=2, num_pairing_params=2, num_outputs=1)
type3 = deepRNN(layers, units, activation_func, num_timesteps=None, num_pairing_params=2, num_outputs=1000)

print("SUCCESS!")


