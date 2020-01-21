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
        self.num_timesteps = num_timesteps
        self.num_pairing_params = num_pairing_params
            
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
        #self.model.summary()
        self.num_params = self.model.count_params()
    
    
    
    def build_partial_RNN(self, layers, units, activation_func, input_len):
        """ builds and returns a partial network with specified structure """
     
        num_layers = len(layers)
        input = Input(shape=(input_len, 1))
        model = input
        
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
                    model = Flatten(input_shape=shape)(model)
                    model = Dense(units[i], activation=activation_func)(model)
                    
                else:
                    model = Dense(units[i], activation=activation_func, input_shape=shape)(model)
            
            elif layers[i] == 's':
                model = SimpleRNN(units[i], activation=activation_func, \
                                  input_shape=shape, return_sequences=return_seq)(model)

            
            elif layers[i] == 'l':
                model = LSTM(units[i], activation=activation_func, \
                             input_shape=shape, return_sequences=return_seq)(model)
            
            elif layers[i] == 'g':
                model = GRU(units[i], activation=activation_func, \
                            input_shape=shape, return_sequences=return_seq)(model)
            
            else:
                raise ValueError('Choose layers from list: Dense (d), \
                                  SimpleRNN (s), LSTM (l), GRU (g).')
            
        return Model(input, model)
        
    
    def build_RNN_one_output(self, layers, units, activation_func, num_timesteps):
        """ type1:  - use preprocessed time series data as inputs.
                    - each sample in data has dimension 'num_timesteps'
                    - predict next number in time series for each sample """
    
        # build RNN with one output
        input = Input(shape=(num_timesteps, 1))
        model = self.build_partial_RNN(layers, units, activation_func, num_timesteps)(input)
        model = Dense(1, activation=activation_func)(model)
        
        self.model = Model(input, model)
        
        
    
    def build_branched_RNN(self, layers, units, activation_func, num_timesteps, num_pairing_params):
        """ type2:  - use preprocessed time series data and three pairing model
                      parameters as inputs to two separate branches
                    - merge branches after data passes through RNN layers
                    - predict next number in time series for each sample """
        
        # data branch
        data_input = Input(shape=(num_timesteps, 1))
        data_branch = self.build_partial_RNN(layers, units, activation_func, num_timesteps)(data_input)
        
        # parameter branch
        param_input = Input(shape=(num_pairing_params,))
        param_branch = Dense(units[-1], activation=activation_func, \
                               input_shape=(num_pairing_params,))(param_input)
        
        # merge branches
        merged = Concatenate()([data_branch, param_branch])
        merged = Dense(1, activation=activation_func)(merged)
        
        # build final model
        self.model = Model([data_input, param_input], merged)
        
        
      
    def build_RNN_vector_output(self, layers, units, activation_func, num_pairing_params, num_outputs):
    
        # build RNN with vector output
        input = Input(shape=(num_timesteps, 1))
        model = self.build_partial_RNN(layers, units, activation_func, num_pairing_params)(input)
        model = Dense(num_outputs, activation=activation_func)(model)
        
        self.model = Model(input, model)

    
    
    def train(self, X, Y, num_epochs, pairing_params=None, use_early_stopping=True):
        """ Train network on a data set X and targets Y for a fixed
            number of epochs or using the EarlyStopping callback. """
    
        if use_early_stopping:
        
            ES = EarlyStopping(monitor='loss', mode='min', \
                               patience=30, verbose=0)
            
            self.fit = self.model.fit(X, Y, epochs=num_epochs, \
                                      callbacks=[ES], verbose=0, \
                                      shuffle=False)
            
        else:
        
            self.fit = self.model.fit(X, Y, epochs=num_epochs, \
                                      verbose=0, shuffle=False)
            
            
    
    def predict(self, X, total_iters):
        """ Use trained model to make prediction on time
            series data set X and extrapolate until total
            number of iterations is reached. """
        
        prediction = np.empty((total_iters,1))
            
        if isinstance(X, list):
        
            train_iters = X[0].shape[0]
            param_input = X[1][0]
            prediction[:train_iters] = self.model.predict(X)
            
            for iter in range(train_iters, total_iters):
                
                data_input = np.empty((1, self.num_timesteps, 1))
                for i in range(self.num_timesteps):
                    data_input[0, i] = prediction[iter-self.num_timesteps+i]
                    
                prediction[iter] = self.model.predict([data_input, param_input])
        
        else:
            train_iters = X.shape[0]
            prediction[:train_iters] = self.model.predict(X)
            
            for iter in range(train_iters, total_iters):
                
                input = np.empty((1, self.num_timesteps, 1))
                for i in range(self.num_timesteps):
                    input[0, i] = prediction[iter-self.num_timesteps+i]
                    
                prediction[iter] = self.model.predict(input)

        return prediction.reshape(-1)
