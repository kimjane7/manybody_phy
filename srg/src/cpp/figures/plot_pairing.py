import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy.stats as stats
import os
import sys
from pylab import *
matplotlib.rcParams['font.family'] = "serif"
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter


srg = np.loadtxt("data/pairing_srg.dat",unpack=True)
magnus = np.loadtxt("data/pairing_srg_magnus.dat",unpack=True)

plt.figure(figsize=(6,6))
fig = plt.figure(1)
axes = plt.gca()
#axes.set_xlim([0,30])
axes.tick_params(labelsize=12)

plt.semilogy(srg[0], srg[1], linewidth=2, color='darkcyan',label='Direct integration')
plt.semilogy(magnus[0], magnus[1], linewidth=2, color='indianred',label='Magnus expansion')


plt.legend(loc=1, shadow=True, fontsize=12)
plt.xlabel(r'Flow Parameter $s$', fontsize=12, weight='normal', family='serif')
plt.ylabel(r'Norm of $H_{od}$', fontsize=12, weight='normal', family='serif')
#plt.title(, fontsize=12, weight='normal', family='serif')
plt.tight_layout()
axes.legend()

figname = 'plot_pairing.png'
plt.savefig(figname, format='png')
os.system('okular '+figname)
plt.clf()