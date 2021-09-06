#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

##python3 hist_by_FARO_pb.py ./fastga_results_all/ ./hist_and_csv/ 100000 1000
#19 histograms by plan F,A ,R O
path=sys.argv[1]
figpath=sys.argv[2]
maxExp=sys.argv[3]
maxEv=sys.argv[4]

hist_pb=[[] for i in range(19)]
name=[]
for fastga in os.listdir(path): #ddir : directory of fastga_plan
    if(fastga in {"fastga_results_planA", "fastga_results_planF","fastga_results_planO"}):
        for plan in os.listdir(os.path.join(path,fastga)):
            #print("maxExp="+str(maxExp)+"_maxEv="+str(maxEv)+"_" in plan,plan,"maxExp="+str(maxExp)+"_maxEv="+str(maxEv))
            #print("maxExp="+str(maxExp)+"_maxEv="+str(maxEv) in plan,plan,"maxExp="+str(maxExp)+"_maxEv="+str(maxEv))
            if("maxExp="+str(maxExp)+"_maxEv="+str(maxEv)+"_" in plan):
                nameid=fastga[-1]
                name.append("plan"+nameid+"_".join(plan.split("_")[1:3]))
                for fastgadir in os.listdir(os.path.join(path,fastga,plan,"raw","data")): #fastgadir : directory of 50 runs of a configuration
                    pb=int(fastgadir.split("_")[0].split("=")[1])
                    average_pb=[]
                    for fname in os.listdir(os.path.join(path,fastga,plan,"raw","data",fastgadir)):
                        with open(os.path.join(path,fastga,plan,"raw","data",fastgadir,fname)) as fd:
                            auc = float(fd.readlines()[0]) *(-1)
                        average_pb.append(auc)
                    if(hist_pb[pb]==[]): #first algo
                        hist_pb[pb].append(average_pb)
                    elif(len(hist_pb[pb])!=len(name)):
                        hist_pb[pb].append(average_pb)
                    else:
                        hist_pb[pb][len(name)-1]+=average_pb #another algo for the same plan

    if("random" in fastga):
        for randir in os.listdir(os.path.join(path,fastga)):
            #eg path: maxEv=100_nbAlgo=15_2021-08-20T1511+0200_results_randoms
            if(("maxEv="+str(maxEv)+"_") in randir):
                #print("maxEv="+str(maxEv) in randir,randir)
                name.append(randir.split("_")[0]+"_random")
                for ddir in os.listdir(os.path.join(path,fastga,randir)): #ddir : directory of one run_elites_all or more
                    if("crossover" in ddir):
                        #name.append("_".join(ddir.split("_")[1:3]))
                        for fastgadir in os.listdir(os.path.join(path,fastga,randir,ddir,"data")): #fastgadir : directory of 50 runs of a configuration
                            average_pb=[]
                            pb=int(fastgadir.split("_")[0].split("=")[1])
                            for fname in os.listdir(os.path.join(path,fastga,randir,ddir,"data",fastgadir)):
                                with open(os.path.join(path,fastga,randir,ddir,"data",fastgadir,fname)) as fd:
                                    auc = float(fd.readlines()[0]) *(-1)
                                average_pb.append(auc)
                            #print(len(hist_pb[pb]),len(name), pb)
                            if(hist_pb[pb]==[]): #first algo
                                #print("entrer random vide")
                                hist_pb[pb].append(average_pb)
                            elif(len(hist_pb[pb])!=len(name)):
                                #print("entrer random !=")
                                hist_pb[pb].append(average_pb)
                            else:
                                hist_pb[pb][len(name)-1]+=average_pb #another algo for the same plan


figdir=os.path.join(figpath,"hist_by_FARO_pb_maxExp={}_maxEv={}".format(maxExp,maxEv))
try:
    os.makedirs(figdir)
except FileExistsError:
    pass
#colors=['yellow', 'green',"blue","pink","purple","orange","magenta","gray","darkred","cyan","brown","olivedrab","thistle","stateblue"]
print(name)
for pb in range(19):
    print(pb, len(hist_pb[pb]))
    for i in hist_pb[pb]:
        print(len(i))
    plt.figure()
    plt.hist(hist_pb[pb],bins=10,range=(0,1),align="mid",rwidth=0.9,edgecolor="red",label=name) #no label color=colors[:len(name)]
    #for aucs in range(len(hist_pb[pb])):
        #plt.hist(hist_pb[pb][aucs],bins=10,range=(0,1),align="mid",rwidth=0.9,edgecolor="red",label=name[aucs]) #no label
    plt.xlabel("performances")
    plt.ylabel("Number of runs")
    plt.ylim(0,800)
    plt.xlim(0,1)
    plt.yticks(range(0,800,50))
    #plt.xticks(np.cumsum([0.1]*10))
    plt.legend()
    plt.savefig(figdir+"/hist_FARO_pb={}_maxExp={}_maxEv={}.png".format(pb,maxExp,maxEv))
    plt.close()