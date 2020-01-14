import numpy as np
from plot import plot_remainder_flow

def grid_search(g, trials, layers_list, units_list, activation_func, \
                timesteps, frac_train, epochs, use_early_stopping=False, \
                plot_loss=True, plot_flow=True)

    for layers in layers_list:
        for units in units_list:
        
            if len(layers)==len(units):
                plot_remainder_flow(g, trials, layers, units, activation_func, timesteps, frac_train, epochs, use_early_stopping)
            else:
                print("Incompatible ")


    
'''
baseline_neural_network(g, trials, layers, tot_units, activation_func, timesteps, frac_train, epochs, use_early_stopping=False, plot_loss=True, plot_flow=True):

make grid search function
good: small loss, small sigma E, close final E prediction and sigma E
    
bound size of search space?
bound total parameters not units?


'''
