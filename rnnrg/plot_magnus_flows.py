import os
import sys
import numpy as np
import matplotlib.pyplot as plt



# plot flow for all g
plt.figure(figsize=(10,8))
fig = plt.figure(1)
axes = plt.gca()
axes.set_xlim([0.0,10.0])
#axes.set_xscale('log')
axes.tick_params(labelsize=12)

g_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
color_list = ['indianred', 'sandybrown', 'orange', 'gold', 'yellowgreen', 'forestgreen', 'mediumturquoise', 'steelblue', 'royalblue', 'mediumpurple']

for g in g_list:

    filename = "data/magnus_g"+str(g)+".dat"
    data = np.loadtxt(filename, unpack=True)

    s = data[0]
    E = data[1]

    plt.plot(s, E, linewidth=2, label='g = '+str(round(g,1)), color=color_list[g_list.index(g)])
    

plt.grid(True, alpha=0.2)
plt.legend(loc=1, ncol=1, fontsize=12)
plt.xlabel("Flow parameter s", fontsize=12)
plt.ylabel("First diagonal element", fontsize=12)
plt.title("IMSRG with Magnus expansion flows for various g", fontsize=12)

figname = "magnus_flows.pdf"
plt.savefig(figname, format='pdf')
os.system('open '+figname)
plt.clf()
