import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import sys
from pylab import *
matplotlib.rcParams['font.family'] = "serif"
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter


plt.figure(figsize=(6,6))
fig = plt.figure(1)
axes = plt.gca()
axes.set_xlim([-1,1])
axes.set_ylim([-0.5,0.05])
axes.tick_params(labelsize=12)

mbpt2 = np.loadtxt("data/Ecorr_pairing_mbpt2.dat",unpack=True)
srg = np.loadtxt("data/Ecorr_pairing_srg.dat",unpack=True)

plt.plot(srg[0], srg[1], color='darkcyan', linewidth=2, label="Exact")
plt.plot(mbpt2[0], mbpt2[1], color='indianred', linewidth=2, label="MBPT2")

plt.legend(loc=1, shadow=True, fontsize=12)
plt.xlabel(r'Interaction Strength $g$', fontsize=12, weight='normal', family='serif')
plt.ylabel(r'Correlation Energy $E_{corr}$', fontsize=12, weight='normal', family='serif')
plt.tight_layout()
plt.grid(True)

figname = 'correlation_energy'+'.png'
plt.savefig(figname, format='png')
os.system('okular '+figname)
plt.clf()
