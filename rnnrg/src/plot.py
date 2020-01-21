import time
import numpy as np
import matplotlib.pyplot as plt
from predict import avg_sigma, predict_remainder_flow, predict_different_flow

def get_label(g, layers, units, activation_func, num_timesteps, transfer_learning=False):
        
    if transfer_learning:
        label = '_gtrain'+str(round(g,1))
    else:
        label = '_g'+str(round(g,1))

    label += '_'+layers
    for unit in units:
        label += '_'+str(unit)
    label += '_'+activation_func+'_timesteps'+str(num_timesteps)
    
    return label
    

def plot_remainder_flow(num_trials, layers, units, activation_func, \
                        g, frac_train, num_epochs, num_timesteps=2, \
                        plot_loss=True, plot_flow=True):
    

    # label for output files
    label = get_label(g, layers, units, activation_func, num_timesteps, False)
    
    # average of predictions
    s_pred, E_pred_avg, E_pred_sigma, s, E, loss_trials, num_params \
                                 = avg_sigma(predict_remainder_flow, num_trials, \
                                             layers, units, activation_func, \
                                             g, frac_train, num_epochs, num_timesteps)
    
    # plot MSE loss trials
    if plot_loss:
    
        # calculate bounds for plot
        max_epoch = 0
        for i in range(num_trials):
            
            final_epoch = len(loss_trials[i])
            if final_epoch > max_epoch:
                max_epoch = final_epoch
        
            epoch_arr = np.arange(1, final_epoch+1)
            plt.semilogy(epoch_arr, loss_trials[i], linewidth=2, alpha=0.1, color='dodgerblue')
            
        plt.title('Loss During Training for d = 1.0, g = '+str(round(g,1)))
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Epoch')
        plt.xlim(1, max_epoch)
        plt.grid(alpha=0.2)
        
        plt.savefig('../figures/remainder'+label+'_loss.pdf', format='pdf')
        plt.clf()
    
    # plot data and predicted flow
    if plot_flow:
    
        plt.plot(s_pred, E_pred_avg, linewidth=2, color='dodgerblue', label='Prediction')
        plt.fill_between(s_pred, E_pred_avg-E_pred_sigma, E_pred_avg+E_pred_sigma, color='dodgerblue', alpha=0.2)
        plt.plot(s, E, linewidth=2, linestyle='--', color='k', label='Data')
        plt.axvspan(0, frac_train*s[-1], alpha=0.2, color='r', label='Training Region')
        
        plt.title('Prediction of IM-SRG Flow for d = 1.0, g = '+str(round(g,1)))
        plt.ylabel('Zero-body Term E(s)')
        plt.xlabel('Flow Parameter s')
        plt.xlim(0, s[-1])
        plt.grid(alpha=0.2)
        plt.legend()
        
        plt.savefig('../figures/remainder'+label+'_flow.pdf', format='pdf')
        plt.clf()
        
    # calculate average and std dev of final loss
    
    loss_avg = 0.0
    loss2_avg = 0.0
    for i in range(num_trials):
        loss = loss_trials[i][-1]
        loss_avg += loss
        loss2_avg += loss**2
    loss_avg /= num_trials
    loss2_avg /= num_trials
    loss_sigma = np.sqrt(loss2_avg-loss_avg**2)
    
    return E[-1], E_pred_avg[-1], E_pred_sigma[-1], loss_avg, loss_sigma, num_params


'''
def plot_different_flow(num_trials, layers, units, activation_func, \
                        g1, g2_list, frac_init, num_epochs, num_timesteps=2, \
                        num_pairing_params=None, plot_loss=True, plot_flow=True):

    # get data
    data1 = get_magnus_data(g1)
    data2 = get_magnus_data(g2)
    s = data1[0]
    E1 = data1[1]
    E2 = data2[1]

    # label for output files
    label = get_label(g, layers, units, activation_func, num_timesteps, num_pairing_params, False)

    # average of predictions
    s_pred, avg_E, sigma_E, loss_trials, num_params \
                                 = avg_sigma(predict_different_flow, num_trials, \
                                             layers, units, activation_func, \
                                             data1, data2, frac_init, num_epochs, \
                                             num_timesteps, num_pairing_params)

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
            

        
        plt.title('Loss During Training for d = 1.0, g = '+str(round(g,1)))
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Epoch')
        plt.xlim(1, max_epoch)
        plt.grid(alpha=0.2)
        
        plt.savefig('../figures/different/'+label+'_loss.pdf', format='pdf')
        plt.clf()

    # plot data and predicted flow
    if plot_flow:

        plt.plot(s_pred, avg_E, linewidth=2, color='dodgerblue', label='Prediction')
        plt.fill_between(s_pred, avg_E-sigma_E, avg_E+sigma_E, color='dodgerblue', alpha=0.2)
        plt.plot(s, E, linewidth=2, linestyle='--', color='k', label='Data')
        plt.axvspan(0, frac_train*s[-1], alpha=0.2, color='r', label='Training Region')
        
        plt.title('Prediction of IM-SRG Flow for d = 1.0, g = '+str(round(g,1)))
        plt.ylabel('Zero-body Term E(s)')
        plt.xlabel('Flow Parameter s')
        plt.xlim(0, s[-1])
        plt.grid(alpha=0.2)
        plt.legend()
        
        plt.savefig('../figures/different/'+label+'_flow.pdf', format='pdf')
        plt.clf()


'''
