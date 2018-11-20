import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import sys
from pylab import *
matplotlib.rcParams['font.family'] = "serif"
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter


def true_gs_energy(s):
	return [1.41677428435]*len(s)

methods = ['srg','srg_magnus','imsrg','imsrg_magnus']
labels = [r'SRG',r'Magnus SRG',r'IMSRG(2)',r'Magnus IMSRG(2)']
energy_col = [2,2,1,1]
offdiag_col = [1,1,5,5]

ds = [0.5,0.1,0.05,0.01,0.005,0.001]
colors = ['indianred','darkorange','yellowgreen','seagreen','dodgerblue','darkviolet']


# make flow plot for each method
for i in range(len(methods)):

	plt.figure(figsize=(6,6))
	fig = plt.figure(1)
	axes = plt.gca()
	axes.set_xlim([0,6])
	axes.set_ylim([1.38,1.52])
	axes.tick_params(labelsize=12)

	# plot energy vs. flowparam for all step sizes
	for j in range(len(ds)):

		filename = '../data/'+methods[i]+'_euler_'+str(ds[j])+'_flow.dat'
		data = np.loadtxt(filename,unpack=True)
		plt.plot(data[0],data[energy_col[i]],linewidth=2,linestyle='-',color=colors[j],label=r'$\delta s =$ '+str(ds[j]))

	# plot analytic ground state energy
	s = np.arange(0,10,0.1)
	plt.plot(s,true_gs_energy(s),linewidth=2,linestyle='--',color='black',label=r'True g.s. energy')

	plt.legend(loc=1, shadow=True, fontsize=12)
	plt.xlabel(r'Flow Parameter $s$', fontsize=12, weight='normal', family='serif')
	plt.ylabel(r'Ground State Energy (MeV)', fontsize=12, weight='normal', family='serif')
	plt.title(labels[i]+r' Flow (Euler)', fontsize=12, weight='normal', family='serif')
	plt.tight_layout()

	figname = methods[i]+'_euler_flow.png'
	plt.savefig(figname, format='png')
	os.system('okular '+figname)
	plt.clf()

