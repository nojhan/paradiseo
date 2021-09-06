#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

##cmd eg : 
# python3 hist_by_2_4_5.py ./fastga_results_all/ ./hist_and_csv/ 100000 1000

#get the Mann Whitney test U results between the plan F and plan R
#  (change ligne 23 and 44 for other plan, and the maxExp, maxEv for other budget)

path=sys.argv[1] 
figpath=sys.argv[2] #directory to store the data
maxExp=sys.argv[3]
maxEv=sys.argv[4]

hist_pb=[[] for i in range(19)]
name=[]
randind=-1
for fastga in os.listdir(path): #ddir : directory of fastga_plan
    if(fastga in {"fastga_results_planF"}):
        for plan in os.listdir(os.path.join(path,fastga)):
            print("maxExp="+str(maxExp)+"_maxEv="+str(maxEv)+"_" in plan,plan,"maxExp="+str(maxExp)+"_maxEv="+str(maxEv))
            #print("maxExp="+str(maxExp)+"_maxEv="+str(maxEv) in plan,plan,"maxExp="+str(maxExp)+"_maxEv="+str(maxEv))
            if("maxExp="+str(maxExp)+"_maxEv="+str(maxEv)+"_" in plan):
                name.append("_".join(plan.split("_")[:3]))
                for fastgadir in os.listdir(os.path.join(path,fastga,plan,"raw","data")): #fastgadir : directory of 50 runs of a configuration
                    pb=int(fastgadir.split("_")[0].split("=")[1])
                    average_pb=[]
                    for fname in os.listdir(os.path.join(path,fastga,plan,"raw","data",fastgadir)):
                        with open(os.path.join(path,fastga,plan,"raw","data",fastgadir,fname)) as fd:
                            auc = float(fd.readlines()[0]) 
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
                print("maxEv="+str(maxEv) in randir,randir)
                name.append(randir.split("_")[0]+"_random")
                randind=len(name)-1
                print(randind,name)
                for ddir in os.listdir(os.path.join(path,fastga,randir)): #ddir : directory of one run_elites_all or more
                    if("crossover" in ddir):
                        for fastgadir in os.listdir(os.path.join(path,fastga,randir,ddir,"data")): #fastgadir : directory of 50 runs of a configuration
                            average_pb=[]
                            pb=int(fastgadir.split("_")[0].split("=")[1])
                            for fname in os.listdir(os.path.join(path,fastga,randir,ddir,"data",fastgadir)):
                                with open(os.path.join(path,fastga,randir,ddir,"data",fastgadir,fname)) as fd:
                                    auc = float(fd.readlines()[0]) 
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


figdir=os.path.join(figpath,"mwtestU_FR")
try:
    os.makedirs(figdir)
except FileExistsError:
    pass
#colors=['yellow', 'green',"blue","pink","purple","orange","magenta","gray","darkred","cyan","brown","olivedrab","thistle","stateblue"]
print(name)

filename="mwtestU_maxExp={}_maxEv={}_FR.csv".format(maxExp,maxEv)
with open(os.path.join(figdir,filename),'w+') as csvfile:
    csvfile.write(" ,"+",".join(map(str,range(0,19)))+"\n")
meanvalue=[]
pvalue=[]
meanR=[]
meanF=[]
mdianR=[]
mdianF=[]
mdianvalue=[]
iqrR=[]
iqrF=[]
stdR=[]
stdF=[]
iqrvalue=[]
pstd=[]

for pb in range(19):
    #hR,lR,_=plt.hist(hist_pb[pb][randind],bins=10,range=(-1,0),align="mid",label=name) #no label color=colors[:len(name)]
    #hF,lF,_=plt.hist(hist_pb[pb][np.abs(1-randind)],bins=10,range=(-1,0),align="mid",label=name) #no label color=colors[:len(name)]
    _,pv=mannwhitneyu(hist_pb[pb][np.abs(1-randind)],hist_pb[pb][randind])
    print(_,pv)
    #meanvalue.append(np.mean(np.array(hF)*np.array(lF[:len(lF)-1]))-np.mean(np.array(hR)*np.array(lR[:len(lR)-1])))
    pstd.append(np.std(hist_pb[pb][np.abs(1-randind)])-np.std(hist_pb[pb][randind]))
    stdF.append(np.std(hist_pb[pb][np.abs(1-randind)]))
    stdR.append(np.std(hist_pb[pb][randind]))
    meanF.append(np.mean(hist_pb[pb][np.abs(1-randind)]))
    meanR.append(np.mean(hist_pb[pb][randind]))
    mdianF.append(np.median(hist_pb[pb][np.abs(1-randind)]))
    mdianR.append(np.median(hist_pb[pb][randind]))
    mdianvalue.append(np.median(hist_pb[pb][np.abs(1-randind)])-np.median(hist_pb[pb][randind]))
    meanvalue.append(np.mean(hist_pb[pb][np.abs(1-randind)])-np.mean(hist_pb[pb][randind]))
    pvalue.append(pv)
    Q1 = np.percentile(hist_pb[pb][np.abs(1-randind)], 25, interpolation = 'midpoint')
    # Third quartile (Q3)
    Q3 = np.percentile(hist_pb[pb][np.abs(1-randind)], 75, interpolation = 'midpoint')
    # Interquaritle range (IQR)
    iqrF.append( Q3 - Q1)
    Q1 = np.percentile(hist_pb[pb][randind], 25, interpolation = 'midpoint')
    # Third quartile (Q3)
    Q3 = np.percentile(hist_pb[pb][randind], 75, interpolation = 'midpoint')
    # Interquaritle range (IQR)
    iqrR.append( Q3 - Q1)
    print(_,pv)
iqrvalue=np.array(iqrF)-np.array(iqrR)
with open(os.path.join(figdir,filename),'a') as csvfile:
    csvfile.write("mF-mR,"+",".join(map(str,meanvalue))+"\n")
    csvfile.write("p_value,"+",".join(map(str,pvalue))+"\n")
    csvfile.write("mF,"+",".join(map(str,meanF))+"\n")
    csvfile.write("mR,"+",".join(map(str,meanR))+"\n")
    csvfile.write("medianF-medianR,"+",".join(map(str,mdianvalue))+"\n")
    csvfile.write("medianF,"+",".join(map(str,mdianF))+"\n")
    csvfile.write("medianR,"+",".join(map(str,mdianR))+"\n")
    csvfile.write("stdF-stdR,"+",".join(map(str,mdianvalue))+"\n")
    csvfile.write("stdF,"+",".join(map(str,stdF))+"\n")
    csvfile.write("stdR,"+",".join(map(str,stdR))+"\n")
    csvfile.write("iqrF,"+",".join(map(str,iqrF))+"\n")
    csvfile.write("iqrR,"+",".join(map(str,iqrR))+"\n")
    csvfile.write("iqrF-iqrR,"+",".join(map(str,iqrvalue))+"\n")

    