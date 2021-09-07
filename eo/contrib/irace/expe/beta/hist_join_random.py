#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

##python3 hist_random.py ./dastga_results_all/fastga_results_random ./hist_and_csv/
#argv : list of elite results
data=sys.argv[1]
figpath=sys.argv[2]
averageConfigs=[]
name=[]
for path in os.listdir(data):
    #eg path: maxEv=100_nbAlgo=15_2021-08-20T1511+0200_results_randoms
    average=[]
    if("maxEv" in path):
        for ddir in os.listdir(os.path.join(data,path)): #ddir : directory of one run_elites_all or more
            if("crossover" in ddir):
                #name.append("_".join(ddir.split("_")[1:3]))
                for fastgadir in os.listdir(os.path.join(data,path,ddir,"data")): #fastgadir : directory of 50 runs of a configuration
                    for fname in os.listdir(os.path.join(data,path,ddir,"data",fastgadir)):
                        with open(os.path.join(data,path,ddir,"data",fastgadir,fname)) as fd:
                            auc = float(fd.readlines()[0]) *(-1)
                        average.append(auc)
                        #hist[belonging(auc,cum)]+=1
        averageConfigs.append(average)
        name.append(path.split("_")[0])

figdir=os.path.join(figpath,"hist_join")
try:
    os.makedirs(figdir)
except FileExistsError:
    pass

colors=['yellow', 'green',"blue","pink","purple","orange","magenta","gray","darkred","cyan","brown","olivedrab","thistle","stateblue"]
plt.figure()
plt.hist(averageConfigs,bins=10,range=(0,1),align="mid",rwidth=0.5,label=name) #no label
plt.xlabel("performances")
plt.ylabel("Number of runs")
plt.ylim([0,8000])
plt.xlim(0,1)
plt.yticks(range(0,8000,500))
#plt.xticks(np.cumsum([0.1]*10))
plt.legend()
plt.savefig(figdir+"/hist_random_by_budget.png")
plt.close()
