import numpy as np
from plot import plot_remainder_flow

def main():

    num_trials = 20
    g_list = [0.1*i for i in range(1, 11)]
    for g in g_list:
        
        # type 1 network
        plot_remainder_flow(g, num_trials, 'ddd', [200, 90, 10], 'linear', 0.1, 500, 5, None, True, True)
        plot_remainder_flow(g, num_trials, 'ddd', [10, 90, 200], 'linear', 0.1, 500, 5, None, True, True)
        plot_remainder_flow(g, num_trials, 'ddd', [100, 100, 100], 'linear', 0.1, 500, 5, None, True, True)



if __name__ == "__main__":
    main()
