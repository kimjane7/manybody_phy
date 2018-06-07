import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import sys
from pylab import *
matplotlib.rcParams['font.family'] = "serif"
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter

g = np.linspace(-1.0, 1.0, num=11)
files = ['data/eigs'+str(i)+'.dat' for i in range(0,11)]
colors = ['indianred','orange', 'yellowgreen', 'seagreen', 'dodgerblue', 'blueviolet']

plt.figure(figsize=(6,6))
fig = plt.figure(1)
axes = plt.gca()
axes.set_xlim([-1,1])
axes.tick_params(labelsize=12)

for i in range(0,11):
	file = np.loadtxt(files[i], unpack=True)
	for j in range(0,6):
		plt.plot(g[i], file[0][j], linewidth=5, marker='o', color=colors[j], alpha=0.6)

plt.xlabel(r'Interaction Strength $g$', fontsize=12, weight='normal', family='serif')
plt.ylabel(r'Eigenvalues', fontsize=12, weight='normal', family='serif')
plt.title(r'Eigenvalues for Energy Spacing $d=1$ and Various $g$', fontsize=12, weight='normal', family='serif')
plt.grid()
plt.tight_layout()

figname = 'eigs.png'
plt.savefig(figname, format='png')
os.system('okular '+figname)
plt.clf()