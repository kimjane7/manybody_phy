import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import sys
from pylab import *
matplotlib.rcParams['font.family'] = "serif"
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter


methods = ['srg','srg_magnus','imsrg','imsrg_magnus']
labels = [r'SRG',r'Magnus SRG',r'IMSRG(2)',r'Magnus IMSRG(2)']
smax = [2.0,2.0,6.0,6.0]
energy_col = [2,2,1,1]
offdiag_col = [1,1,5,5]
offdiag_labels = [r'$\| H_{od} \|$',r'$\| H_{od} \|$',r'$\| \Gamma_{od} \|$',r'$\| \Gamma_{od} \|$']


ds = [0.5,0.1,0.05,0.01,0.005,0.001]
colors = ['indianred','darkorange','yellowgreen','seagreen','dodgerblue','darkviolet']


# make flow plots for each method
for i in range(len(methods)):


	#############################
	########## E vs. s ##########
	#############################

	'''
	plt.figure(figsize=(6,6))
	fig = plt.figure(1)
	axes = plt.gca()
	axes.set_xlim([0,smax[i]])
	axes.set_ylim([1.38,1.52])
	axes.tick_params(labelsize=12)

	# plot energy vs. flowparam for all step sizes
	for j in range(len(ds)):

		filename = '../data/'+methods[i]+'_euler_'+str(ds[j])+'_flow.dat'
		data = np.loadtxt(filename,unpack=True)
		plt.plot(data[0],data[energy_col[i]],linewidth=2,linestyle='-',color=colors[j],label=r'$\delta s =$ '+str(ds[j]))

	# plot analytic ground state energy
	s = np.arange(0,10,0.1)
	plt.plot(s,[1.41677428435]*len(s),linewidth=2,linestyle='--',color='black',label=r'True g.s. energy')

	plt.legend(loc=1, shadow=True, fontsize=12)
	plt.xlabel(r'Flow Parameter $s$', fontsize=12, weight='normal', family='serif')
	plt.ylabel(r'Ground State Energy (MeV)', fontsize=12, weight='normal', family='serif')
	plt.title(labels[i]+r' Flow (Euler)', fontsize=12, weight='normal', family='serif')
	plt.tight_layout()

	figname = methods[i]+'_euler_energy.png'
	plt.savefig(figname, format='png')
	os.system('okular '+figname)
	plt.clf()
	'''



	##################################################
	########## ||Hod|| or ||Gammaod|| vs. s ##########
	##################################################

	plt.figure(figsize=(6,6))
	fig = plt.figure(1)
	axes = plt.gca()
	axes.set_xlim([1e-2,10])
	#axes.set_ylim([1e-6,2])
	axes.tick_params(labelsize=12)

	for j in range(len(ds)):

		filename = '../data/'+methods[i]+'_euler_'+str(ds[j])+'_flow.dat'
		data = np.loadtxt(filename,unpack=True)

		# remove values out of range for log plot
		flowparam = []
		offdiag = []
		for k in range(len(data[0])):
			s = data[0][k]
			value = data[offdiag_col[i]][k]
			if (0.0 < s < 10.0) and (0.0 < value < 2):
				flowparam.append(s)
				offdiag.append(value)

		plt.semilogy(flowparam,offdiag,linewidth=2,linestyle='-',color=colors[j],label=r'$\delta s =$ '+str(ds[j]))


	plt.legend(loc=1, shadow=True, fontsize=12)
	plt.xlabel(r'Flow Parameter $s$', fontsize=12, weight='normal', family='serif')
	plt.ylabel(offdiag_labels[i], fontsize=12, weight='normal', family='serif')
	plt.title(labels[i]+r' Flow (Euler)', fontsize=12, weight='normal', family='serif')
	plt.tight_layout()

	figname = methods[i]+'_euler_offdiag.png'
	plt.savefig(figname, format='png')
	os.system('okular '+figname)
	plt.clf()
