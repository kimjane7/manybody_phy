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


imsrg = np.loadtxt("data/pairing_imsrg.dat",unpack=True)

plt.figure(figsize=(5,5))
fig = plt.figure(1)
axes = plt.gca()
#axes.set_xlim([0,30])
axes.tick_params(labelsize=12)

plt.semilogx(imsrg[0], imsrg[1], linewidth=2, color='darkcyan')

plt.legend(loc=1, shadow=True, fontsize=12)
plt.xlabel(r'Flow Parameter $s$', fontsize=12, weight='normal', family='serif')
plt.ylabel(r'$E$', fontsize=12, weight='normal', family='serif')
#plt.title(r'$E$', fontsize=12, weight='normal', family='serif')
plt.tight_layout()
axes.legend()

figname = 'imsrg_flow.png'
plt.savefig(figname, format='png')
os.system('okular '+figname)
plt.clf()