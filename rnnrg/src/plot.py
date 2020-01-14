import time
import numpy as np
import matplotlib.pyplot as plt
from predict import avg_sigma, predict_remainder_flow

def plot_remainder_flow(g, num_trials, layers, units, activation_func, \
                        num_timesteps, frac_train, num_epochs, \
                        use_early_stopping=False, plot_loss=True, plot_flow=True):

    # get data
    datafile = "../data/magnus_ds0.01_g"+str(g)+".dat"
    data = np.loadtxt(datafile, unpack=True)
    s = data[0]
    E = data[1]

    # label for output files
    units_str = ''
    for unit in units:
        units_str += '_'+str(unit)
    name = layers+units_str+'_'+str(num_timesteps)+'_'+str(g)+'_'+str(activation_func)
    
    # average of predictions
    s_pred, avg_E, sigma_E, loss_trials, num_params \
                                 = avg_sigma(predict_remainder_flow, num_trials, \
                                             layers, units, activation_func, \
                                             data, frac_train, num_timesteps, \
                                             num_epochs, use_early_stopping)
    
    
    # calculate avg num training epochs, avg and sigma loss
    max_epoch = 0
    final_losses = np.empty(num_trials)
    final_epochs = np.empty(num_trials)
    
    
    
    # plot MSE loss trials
    if plot_loss:
    
        
        
        for i in range(num_trials):
        
            final_losses[i] = loss_trials[i][-1]
            final_epochs[i] = len(loss_trials[i])
            if final_epochs[i] > max_epoch:
                max_epoch = final_epochs[i]
        
            epoch_arr = np.arange(1, final_epochs[i]+1)
            plt.semilogy(epoch_arr, loss_trials[i], linewidth=2, alpha=0.1, color='dodgerblue')
            

        
        plt.title('Loss During Training for d = 1.0, g = '+str(g))
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Epoch')
        plt.xlim(1, max_epoch)
        plt.grid(alpha=0.2)
        
        plt.savefig('../figures/'+name+'_loss.pdf', format='pdf')
        plt.clf()
    
    # plot data and predicted flow
    if plot_flow:
    
        plt.plot(s_pred, avg_E, linewidth=2, color='dodgerblue', label='Prediction')
        plt.fill_between(s_pred, avg_E-sigma_E, avg_E+sigma_E, color='dodgerblue', alpha=0.2)
        plt.plot(s, E, linewidth=2, linestyle='--', color='k', label='Data')
        plt.axvspan(0, frac_train*s[-1], alpha=0.2, color='r', label='Training Region')
        
        plt.title('Prediction of IM-SRG Flow for d = 1.0, g = '+str(g))
        plt.ylabel('Zero-body Term E(s)')
        plt.xlabel('Flow Parameter s')
        plt.xlim(0, s[-1])
        plt.grid(alpha=0.2)
        plt.legend()
        
        plt.savefig('../figures/'+name+'_flow.pdf', format='pdf')
        plt.clf()
    

        

 # predict different flow
