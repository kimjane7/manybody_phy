import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from predict import predict_flow


def compare_activation_funcs(fixed_units):

    # fixed parameters
    g = 0.1
    frac_train = 0.05
    num_layers=1
    step=2

    # activation functions
    funcs = ['linear', 'elu', 'relu', 'tanh']
    
    # make plot
    fig, ax = plt.subplots(1, 2, figsize=(16,8), sharey=True)
    colors = ['firebrick', 'darkorange', 'seagreen', 'royalblue']

    # train RNNs and plot predictions
    for i in range(len(funcs)):
    
        # simple RNN
        prediction, data = predict_flow(g, frac_train, 'simple', 1, [fixed_units], funcs[i], step)
        ax[0].plot(prediction[0], prediction[1], color=colors[i], linewidth=5, label=funcs[i])
        
        # LSTM
        prediction, data = predict_flow(g, frac_train, 'lstm', 1, [fixed_units], funcs[i], step)
        ax[1].plot(prediction[0], prediction[1], color=colors[i], linewidth=5, label=funcs[i])
                                
    ax[0].plot(data[0], data[1], color='k', linewidth=2, linestyle='dashed', label='data')
    ax[0].axvspan(0.0, frac_train*data[0][-1], alpha=0.5, color='red', label='training region')
    ax[0].set_xlim(data[0][0],data[0][-1])
    ax[0].set_xlabel(r'flow parameter $s$')
    ax[0].set_ylabel(r'first diagonal element of $\hat{H}$')
    ax[0].title.set_text('simple RNN')
    
    ax[1].plot(data[0], data[1], color='k', linewidth=2, linestyle='dashed', label='data')
    ax[1].axvspan(0.0, frac_train*data[0][-1], alpha=0.5, color='red', label='training region')
    ax[1].set_xlim(data[0][0],data[0][-1])
    ax[1].set_xlabel(r'flow parameter $s$')
    ax[1].legend(loc=1)
    ax[1].title.set_text('LSTM')
    
    plt.suptitle('RNN predictions using '+str(fixed_units)+' units', fontsize=16)
    fig.savefig('compare_activation_funcs_'+str(fixed_units)+'.png')


"""
# plot all flows
plt.subplot(1,2,1)
plt.plot(s, E, label='data')
for i in range(len(flows)):
    plt.plot(s_train, flows[i], color=colors[i], label=str(units1[i])+' units')

# plot eigenvalue predictions as function of units
plt.subplot(1,2,2)
plt.hlines(E[-1], units1[0], units1[-1])
for i in range(len(flows)):
    plt.plot(units1[i], flows[i][-1], color=colors[i], marker='o', markersize=10)

plt.show()
"""
