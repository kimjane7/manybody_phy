import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import sys
from pylab import *
matplotlib.rcParams['font.family'] = "serif"
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter

plt.figure(figsize=(8,6))
fig = plt.figure(1)
axes = plt.gca()
axes.tick_params(labelsize=12)

max_variation = [20]
MC_cycles = [10000]

for i in max_variation:
	for j in MC_cycles:

		name = "data/bosons_"+str(i)+"_"+str(j)
		data = np.loadtxt(name+".dat",unpack=True)
		X,Y = np.meshgrid(np.unique(data[0]),np.unique(data[1]))
		Z = data[2].reshape((i,i))

		cs = plt.contourf(X,Y,Z,20,cmap='plasma',vmin=0.0,vmax=6.5)
		m = plt.cm.ScalarMappable(cmap='plasma')
		m.set_array(Z)
		m.set_clim(0.0,6.5)
		plt.colorbar(m,boundaries=np.arange(0.0,6.5,0.5))

		plt.xlabel(r'$\alpha$', fontsize=12, weight='normal', family='serif')
		plt.ylabel(r'$\beta$', fontsize=12, weight='normal', family='serif')
		plt.title(r'Ground State Energy ('+str(j)+' MC cycles)', fontsize=12, weight='normal', family='serif')
		plt.tight_layout()

		plt.savefig(name+".png", format='png')
		os.system('okular '+name+".png")
		plt.clf()

