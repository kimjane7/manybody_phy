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


file = np.loadtxt("data/magnus_flow.dat",unpack=True)


# E
ax1 = plt.subplot(313)
plt.plot(file[0], file[1],color='darkcyan',linewidth=2)
plt.setp(ax1.get_xticklabels(), fontsize=12)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.ylabel(r'$E$', fontsize=12, weight='normal', family='serif')
plt.xlabel(r'Flow Parameter $s$', fontsize=12, weight='normal', family='serif')


# dE
ax2 = plt.subplot(312, sharex=ax1)
plt.plot(file[0], file[2],color='indianred',linewidth=2)
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.ylabel(r'$\frac{dE}{ds}$', fontsize=12, weight='normal', family='serif')

# ||Gammaod||
ax3 = plt.subplot(311, sharex=ax1)
plt.plot(file[0], file[3],color='forestgreen',linewidth=2)
plt.setp(ax3.get_xticklabels(), visible=False)
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.ylabel(r'$||\Gamma_{od}||$', fontsize=12, weight='normal', family='serif')


plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
figname = 'magnus_flow.png'
plt.savefig(figname, format='png')
os.system('okular '+figname)
plt.clf()