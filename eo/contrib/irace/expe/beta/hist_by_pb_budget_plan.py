#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# python3 hist_by_pb_budget_plan.py ./fastga_results_all/fastga_results_planF/ ./hist_and_csv/
#python3 hist_by_pb_budget_plan.py ./fastga_results_all/fastga_results_planO ./hist_and_csv
#get 19 histograms with number of budget bars, same as hist_join but now is by pb

#argv : list of elite results
path=sys.argv[1]
figpath=sys.argv[2]
#plan_name=sys.argv[3]
hist_pb=[[] for i in range(19)]
name=[]
if("random" in path):
    plan_name="R"
else:
    plan_name=path.strip("/").split("/")[-1][-1]


for plandir in os.listdir(path): #plandir: directory of an experiment of elite results 
    if("results_elites_all" in plandir):
        #eg : plan2_maxExp=10000_maxEv=1000_2021-08-20T1347+0200_results_elites_all
        budget_irace=plandir.split("_")[1].split("=")[1]
        budget_fastga=plandir.split("_")[2].split("=")[1]
        name.append("plan="+plan_name+"_"+"".join(plandir.split("_")[1:3])) #plan=*_maxExp=*_maxEv=*
        
        for algodir in os.listdir(os.path.join(path,plandir,"raw","data")):
            average_pb=[]
            pb=int(algodir.split("_")[0].split("=")[1])
            for algo in os.listdir(os.path.join(path,plandir,"raw","data",algodir)):
                with open(os.path.join(path,plandir,"raw","data",algodir,algo)) as fd:
                    auc = float(fd.readlines()[0]) *(-1)
                average_pb.append(auc)
            if(hist_pb[pb]==[]): #first algo
                hist_pb[pb].append(average_pb)
            elif(len(hist_pb[pb])!=len(name)):
                hist_pb[pb].append(average_pb)
            else:
                hist_pb[pb][len(name)-1]+=average_pb #another algo for the same plan

    if("results_randoms" in plandir):
        #eg : maxEv=1000_2021-08-20T1347+0200_results_random
        budget_fastga=plandir.split("_")[0].split("=")[1]
        name.append("plan="+plan_name+"_"+"".join(plandir.split("_")[0])) #plan=*_maxExp=*_maxEv=*
        for algodir in os.listdir(os.path.join(path,plandir)):
            
            for algo in os.listdir(os.path.join(path,plandir,algodir,"data")):
                pb=int(algo.split("_")[0].split("=")[1])
                average_pb=[]
                for fname in os.listdir(os.path.join(path,plandir,algodir,"data",algo)):
                    with open(os.path.join(path,plandir,algodir,"data",algo,fname)) as fd:
                        auc = float(fd.readlines()[0]) *(-1)
                    average_pb.append(auc)
                if(hist_pb[pb]==[]): #first algo
                    print("entrer")
                    hist_pb[pb].append(average_pb)
                elif(len(hist_pb[pb])!=len(name)):
                    hist_pb[pb].append(average_pb)
                else:
                    hist_pb[pb][len(name)-1]+=average_pb #another algo for the same plan
        


print(path.split("/")[-1][-1])

figdir=os.path.join(figpath,"hist_by_{}_pb_budget_plan".format(plan_name))
#figdir=os.path.join(figpath,"hist_by_{}_pb_irace_maxEv={}".format(plan_name,1000))
try:
    os.makedirs(figdir)
except FileExistsError:
    pass


for pb in range(19):
    print(pb, len(hist_pb[pb]))
    plt.figure()
    plt.hist(hist_pb[pb],bins=10,range=(0,1),align="mid",rwidth=0.9,edgecolor="red",label=name) #no label color=colors[:len(name)]
    #for aucs in range(len(hist_pb[pb])):
        #plt.hist(hist_pb[pb][aucs],bins=10,range=(0,1),align="mid",rwidth=0.9,edgecolor="red",label=name[aucs]) #no label
    plt.xlabel("performances")
    plt.ylabel("Number of runs")
    plt.ylim(0,750)
    plt.yticks(range(0,750,50))
    plt.xlim(0,1)
    plt.legend()
    plt.savefig(figdir+"/hist_plan={}_pb={}_budget.png".format(plan_name,pb))
    plt.close()