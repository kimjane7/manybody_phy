import numpy as np
from plot import plot_remainder_flow


def grid_search(num_trials, layers_list, units_list, activation_func_list, g, \
                frac_train_list, num_epochs, timesteps_list, plot_loss=True, plot_flow=True)

    for layers in layers_list:
        for units in units_list:
            for activation_func in activation_func_list:
                for frac_train in frac_train_list:
                    for timesteps in timesteps_list:
        
                        E, E_pred_avg, E_pred_sigma, loss_avg, loss_sigma, num_params \
                                   = plot_remainder_flow(num_trials, layers, units, \
                                                         activation_func, g, frac_train, \
                                                         num_epochs, num_timesteps, \
                                                         plot_loss, plot_flow):
