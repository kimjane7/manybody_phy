import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from predict import predict_flow


def sidebyside_predictions(units_list, figname):
    """Compare predicted flows using different activation functions.
       Plot for simple RNN and LSTM side by side.
       Shape of RNNs given by units in units_list.
       Save to sidebyside_figname_shape.png.
       
       After individual plots are made, summarize for entire
       units_list in plot saved to sidebyside_figname_summary.png.
       """

    # fixed parameters
    g = 0.5
    frac_train = 0.05
    step=2

    # activation functions
    funcs = ['linear', 'relu', 'tanh']
    colors = ['firebrick', 'darkorange', 'royalblue']
    
    # container for results
    predictions_simple = np.empty((len(units_list),len(funcs)))
    predictions_lstm = np.empty((len(units_list),len(funcs)))
    
    # make plot for each units
    for i in range(len(units_list)):
    
        # make plot
        fig, ax = plt.subplots(1, 2, figsize=(15,6), sharey=True)

        # train RNNs and plot predictions
        for j in range(len(funcs)):
        
            # simple RNN
            predictions, data = predict_flow(g, frac_train, 'simple', units_list[i], funcs[j], step)
            predictions_simple[i,j] = predictions[1][-1]
            ax[0].plot(predictions[0], predictions[1], color=colors[j], alpha=0.9, linewidth=5, label=funcs[j])
            
            # LSTM
            predictions, data = predict_flow(g, frac_train, 'lstm', units_list[i], funcs[j], step)
            predictions_lstm[i,j] = predictions[1][-1]
            ax[1].plot(predictions[0], predictions[1], color=colors[j], alpha=0.9, linewidth=5, label=funcs[j])
        
        # simple RNN
        ax[0].plot(data[0], data[1], color='k', linewidth=2, linestyle='dashed', label='data')
        ax[0].axvspan(0.0, frac_train*data[0][-1], alpha=0.2, color='red', label='training region')
        ax[0].set_xlim(data[0][0],data[0][-1])
        ax[0].set_xlabel(r'flow parameter $s$')
        ax[0].set_ylabel(r'first diagonal element of $\hat{H}(s)$')
        ax[0].title.set_text('simple RNN')
        
        # LSTM
        ax[1].plot(data[0], data[1], color='k', linewidth=2, linestyle='dashed', label='data')
        ax[1].axvspan(0.0, frac_train*data[0][-1], alpha=0.2, color='red', label='training region')
        ax[1].set_xlim(data[0][0],data[0][-1])
        ax[1].set_xlabel(r'flow parameter $s$')
        ax[1].legend(loc=1)
        ax[1].title.set_text('LSTM')
        
        if len(units_list[i])==1:
            plt.suptitle('RNN predictions for $g=$'+str(g)+' using '+str(units_list[i][0])+' units', fontsize=16)
        else:
            plt.suptitle('deep RNN predictions for $g=$'+str(g)+' using '+str(units_list[i])+' units', fontsize=16)
        
        shape = ''
        for unit in units_list[i]:
            shape += '_'+str(unit)
        fig.savefig('sidebyside_'+figname+shape+'.png')
        plt.clf()
        
    # summary plot of residuals after flow
    fig, ax = plt.subplots(1, 2, figsize=(15,8), sharey=True)
    residuals_simple = predictions_simple-data[1][-1]
    residuals_lstm = predictions_lstm-data[1][-1]
    
    x = np.arange(len(units_list))
    for j in range(len(funcs)):
        ax[0].plot(x, residuals_simple[:,j], marker='o', color=colors[j], label=funcs[j])
        ax[1].plot(x, residuals_lstm[:,j], marker='o', color=colors[j], label=funcs[j])
        
    ax[0].grid()
    ax[0].title.set_text('simple RNN')
    ax[0].set_xlabel(r'shape of RNN')
    ax[0].set_ylabel(r'residuals')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(units_list, rotation=45)
    
    ax[1].grid()
    ax[1].legend(loc=1)
    ax[1].title.set_text('LSTM')
    ax[1].set_xlabel(r'shape of RNN')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(units_list, rotation=45)
    
    plt.suptitle('RNN predictions for $g=$'+str(g), fontsize=16)
    fig.savefig('sidebyside_'+figname+'_summary.png')
    plt.clf()
