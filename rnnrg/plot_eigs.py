import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot():

    # command line inputs
    method = sys.argv[1]
    activation_func = sys.argv[2]
    s_train = float(sys.argv[3])
    step = int(sys.argv[4])

    # only train for s < 2.5 b/c data is evenly spaced
    s_train = min(s_train,2.5)
    
    # get results and extract data
    results_file = "results/"+method+"_"+activation_func+"_train"+str(round(s_train,2))+"_step"+str(step)+".dat"
    results = np.loadtxt(results_file)
    """
    
    plt.figure(figsize=(10,8))
    fig = plt.figure(1)
    axes = plt.gca()
    axes.set_xlim([s[0],2.0*s_train])
    axes.set_xscale('log')
    axes.tick_params(labelsize=12)

    if num_eigs > 1:
    
        plt.plot(s, eigs[:,0], linewidth=2, label="data", color='orange', marker='o')
        plt.plot(s[step:-step], prediction_rnn[:,0],linewidth=1, label="RNN prediction", color='red')
        plt.plot(s[step:-step], prediction_lstm[:,0],linewidth=1, label="LSTM prediction", color='blue')
        
        for j in range(1,num_eigs):
            plt.plot(s, eigs[:,j], linewidth=2, color='orange', marker='o')
            plt.plot(s[step:-step], prediction_rnn[:,j],linewidth=1, color='red')
            plt.plot(s[step:-step], prediction_lstm[:,j],linewidth=1, color='blue')
    else:
        train_params, cov = curve_fit(exponential, s[:num_pts_train], eigs[:num_pts_train], p0=(1,2,1))
        train_exp_fit = exponential(s, *train_params)
        plt.plot(s, eigs, linewidth=5, label="data", color='steelblue')

    
    
    plt.axvspan(0.0, s[num_pts_train], color='red', alpha=0.2, label="training region")
    plt.grid(True, alpha=0.2)
    plt.legend(loc=1, ncol=1, fontsize=12)
    plt.xlabel("Flow parameter s", fontsize=12)
    plt.ylabel("Eigenvalues", fontsize=12)
    plt.title("Eigenvalue prediction by RNN and LSTM", fontsize=12)

    figname = "eigenvalue_prediction.pdf"
    plt.savefig(figname, format='pdf')
    os.system('open '+figname)
    plt.clf()

    ##########################################
    ####### residuals after whole flow #######
    ##########################################

    '''
    if num_eigs > 1:
    
        plt.figure(figsize=(10,8))
        fig = plt.figure(1)
        axes = plt.gca()
        axes.set_xlim([0,10])
        #axes.set_xscale('log')
        axes.tick_params(labelsize=12)
        
    
    
    print(eigs[-1]-prediction_rnn[-1])
    print(eigs[-1]-prediction_lstm[-1])
    '''
    """
    


if __name__ == "__main__":
    main()
