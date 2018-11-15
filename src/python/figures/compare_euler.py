import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import sys
from pylab import *
matplotlib.rcParams['font.family'] = "serif"
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter

pops = ['A','B','C','D']
a = [r'a = 4',r'a = 4',r'a = 4',r'a = 4']
b = [r'b = 1',r'b = 2',r'b = 3',r'b = 4']
c = [r'c = 0.5',r'c = 0.5',r'c = 0.5',r'c = 0.5']
ds = [0.01, 0.1, 0.5, 1.0]

labels = ['Susceptible','Infected','Resistant']
colors = ['yellowgreen','indianred','dodgerblue']

plt.figure(figsize=(6,6))
fig = plt.figure(1)
axes = plt.gca()
axes.set_xlim([0,10])
axes.set_ylim([0,2])
axes.tick_params(labelsize=12)


for i in range(0,4):

	srg = "../data/srg_euler_"+str(j)+"_flow.dat"
	srg_mag = "../data/srg_magnus_euler_"+str(j)+"_flow.dat"
	imsrg = "../data/imsrg_euler_"+str(j)+"_flow.dat"
	imsrg_mag = "../data/imsrg_magnus_euler_"+str(j)+"_flow.dat"

	plt.plot
	
	plt.plot(stats[0],stats[1],linewidth=1,linestyle='-',color=colors[0],label=labels[0])
	plt.plot(stats[0],stats[3],linewidth=1,linestyle='-',color=colors[1],label=labels[1])
	plt.plot(stats[0],stats[5],linewidth=1,linestyle='-',color=colors[2],label=labels[2])

	plt.plot(stats[0],stats[1],linewidth=1,linestyle='-',color='k',label='Average')
	plt.plot(stats[0],stats[3],linewidth=1,linestyle='-',color='k')
	plt.plot(stats[0],stats[5],linewidth=1,linestyle='-',color='k')


	plt.legend(loc=1, shadow=True, fontsize=12)
	plt.xlabel(r'Time', fontsize=12, weight='normal', family='serif')
	plt.ylabel(r'Number of People', fontsize=12, weight='normal', family='serif')
	plt.title(r'SIRS Model for Population '+pops[i]+' (100 Samples)', fontsize=12, weight='normal', family='serif')
	#plt.grid()
	plt.tight_layout()

	figname = 'trials_'+pops[i]+'100.png'
	plt.savefig(figname, format='png')
	os.system('okular '+figname)
	plt.clf()


