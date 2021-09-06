#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

#cmd : python3 hist_join.py ./fastga_results_all/fastga_results_planO/ ./hist_and_csv/
#histogram by plan for the budgets (irace and fastag)


path=sys.argv[1] #argv : directory of a Plan (O, A, F)
figpath=sys.argv[2] #path to store the histograms
averageConfigs=[]
name=[]
if("fastga_results_plan" in path):
    for ddir in os.listdir(path): #ddir : directory of one run_elites_all or more
        if("plan" in ddir):
            average=[]
            name.append("_".join(ddir.split("_")[1:3]))
            for fastgadir in os.listdir(os.path.join(path,ddir,"raw","data")): #fastgadir : directory of 50 runs of a configuration
                for fname in os.listdir(os.path.join(path,ddir,"raw","data",fastgadir)):
                    with open(os.path.join(path,ddir,"raw","data",fastgadir,fname)) as fd:
                        auc = float(fd.readlines()[0]) *(-1)
                    average.append(auc)
                    #hist[belonging(auc,cum)]+=1
            averageConfigs.append(average)
                #print(hist)
                #print(average)

    figdir=os.path.join(figpath,"hist_join")
    try:
        os.makedirs(figdir)
    except FileExistsError:
        pass


    print(name,len(averageConfigs))

    """
    idd0=name[0].split("_")[0].split("=")[1][:-3]+"k"
    idd1=name[1].split("_")[0].split("=")[1][:-3]+"k"
    idd2=name[2].split("_")[0].split("=")[1][:-3]+"k"

    #only for Budget irace 10000, 50000, 100000 ie: only three experiment results
    titlename="median"+idd0+"={:.3f}".format(np.median(averageConfigs[0]))+" , median"+idd1+"={:.3f}".format(np.median(averageConfigs[1]))+" , median"+idd2+"={:.3f}".format(np.median(averageConfigs[2]))
    _,pv=mannwhitneyu(averageConfigs[0],averageConfigs[1])
    titlename+="\n pvalue{}={:.3f}".format(idd0+idd1,pv)
    _,pv=mannwhitneyu(averageConfigs[0],averageConfigs[2])
    titlename+=" ,pvalue{}={:.3f}".format(idd0+idd2,pv)
    _,pv=mannwhitneyu(averageConfigs[1],averageConfigs[2])
    titlename+=" ,pvalue{}={:.3f}".format(idd1+idd2,pv)
    print(titlename)
    """
    plt.figure()
    plt.hist(averageConfigs,bins=10,range=(0,1),align="mid",rwidth=0.9,label=name) #no label
    plt.xlabel("performances")
    plt.ylabel("Number of runs")
    plt.xlim(0,1)
    plt.ylim(0,7000)
    plt.yticks(range(0,7000,500))
    #plt.title(titlename)
    plt.legend()
    plt.savefig(figdir+"/hist_plan"+path.strip("/")[-1]+"_by_budget.png")
    #plt.savefig(figpath+"/hist_plan"+path.strip("/")[-1]+"_by_budgetI.png")
    plt.close()
    

