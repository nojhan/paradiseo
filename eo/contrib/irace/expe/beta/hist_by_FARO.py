#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

##python3 hist_by_FARO.py ./fastga_results_all/ ./hist_and_csv/ 100000 1000
#one plot for one experiment plan for the same budget fastga, and the same budget irace if there is a budget irace (A,F)
path=sys.argv[1]
figpath=sys.argv[2]
maxExp=sys.argv[3]
maxEv=sys.argv[4]

indF=-1
indFO=-1
averageConfigs=[]
name=[]
for fastga in os.listdir(path): #ddir : directory of fastga_plan
    if(fastga in {"fastga_results_planA","fastga_results_planF","fastga_results_planO"}):
        for plan in os.listdir(os.path.join(path,fastga)):
            print("maxExp="+str(maxExp)+"_maxEv="+str(maxEv) in plan,plan,"maxExp="+str(maxExp)+"_maxEv="+str(maxEv))
            if("maxExp="+str(maxExp)+"_maxEv="+str(maxEv) in plan):
                average=[]
                
                for fastgadir in os.listdir(os.path.join(path,fastga,plan,"raw","data")): #fastgadir : directory of 50 runs of a configuration
                    for fname in os.listdir(os.path.join(path,fastga,plan,"raw","data",fastgadir)):
                        with open(os.path.join(path,fastga,plan,"raw","data",fastgadir,fname)) as fd:
                            auc = float(fd.readlines()[0]) *(-1)
                        average.append(auc)
                averageConfigs.append(average)
                nameid=plan.split("_")[0][-1]
                name.append("plan"+nameid+"_"+"_".join(plan.split("_")[1:3]))
    if("random" in fastga):
        for randir in os.listdir(os.path.join(path,fastga)):
            #eg path: maxEv=100_nbAlgo=15_2021-08-20T1511+0200_results_randoms
            average=[]
            if("maxEv="+str(maxEv)+"_" in randir):
                for ddir in os.listdir(os.path.join(path,fastga,randir)): #ddir : directory of one run_elites_all or more
                    if("crossover" in ddir):
                        #name.append("_".join(ddir.split("_")[1:3]))
                        for fastgadir in os.listdir(os.path.join(path,fastga,randir,ddir,"data")): #fastgadir : directory of 50 runs of a configuration
                            for fname in os.listdir(os.path.join(path,fastga,randir,ddir,"data",fastgadir)):
                                with open(os.path.join(path,fastga,randir,ddir,"data",fastgadir,fname)) as fd:
                                    auc = float(fd.readlines()[0]) *(-1)
                                average.append(auc)
                                #hist[belonging(auc,cum)]+=1
                averageConfigs.append(average)
                name.append(randir.split("_")[0]+"_random")


figdir=os.path.join(figpath,"hist_FARO_by_budget")
try:
    os.makedirs(figdir)
except FileExistsError:
    pass

#_,pv=mannwhitneyu(averageConfigs[indFO],averageConfigs[indF])
#print(name,len(averageConfigs))
plt.figure()
plt.hist(averageConfigs,bins=10,range=(0,1),align="mid",rwidth=0.9,label=name) #no label
plt.xlabel("performances")
plt.ylabel("Number of runs")
plt.xlim(0,1)
plt.ylim(0,8000)
plt.yticks(range(0,8000,500))
#plt.title("pvalue="+str(pv)+"\n medianeF="+str(np.median(averageConfigs[indF]))+", medianeFO="+str(np.median(averageConfigs[indFO])))
plt.legend()
plt.savefig(figdir+"/hist_planFARO"+"_maxExp="+str(maxExp)+"_maxEv="+str(maxEv)+".png")
plt.close()
    