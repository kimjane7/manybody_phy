import numpy as np
from plot import plot_remainder_flow

def main():

    trials = 30
    g_list = [round(0.1*i, 1) for i in range(1,11)]
    for g in g_list:
        plot_remainder_flow(g, trials, 'ddd', [100, 100, 100], 'linear', 5, 0.1, 1000, True)
        plot_remainder_flow(g, trials, 'ddd', [10, 90, 200], 'linear', 5, 0.1, 1000, True)
        plot_remainder_flow(g, trials, 'ddd', [200, 90, 10], 'linear', 5, 0.1, 1000, True)



if __name__ == "__main__":
    main()
