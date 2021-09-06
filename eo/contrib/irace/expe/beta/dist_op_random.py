#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

#cmd eg : python3 ./dist_op_random.py ./fastga_results_all/fastga_results_random/ ./hist_and_csv/
#get the distribution of operators variants recommended by 15 random algo for each maxEv 
#pc and pm 10 possibilities :   [0-0.1[  [0.1-0.2[ [0.2-0.3[  [0.3-0.4[  [0-0.5[  [0.5-0.6[ ...[0.9-1[
#pop-size and offspring-size 10 possibilities : 0-5 5-10, 10-15 15-20  20-25  25-30 30-35- 35-40 40-45 45-50

path=sys.argv[1] # directory of a result of one experiment
#eg : ./fastga_results_all/fastga_results_random/
figdir=sys.argv[2] #directory of where you want to store the data
if("random" in path):
    #column : [operator : nbpossibilities]
    distdir=figdir+"/distribution_random"
    try:
        os.makedirs(distdir)
    except FileExistsError:
        pass

    nbparam=9 #-1 car il y a le pb

    res=[]
            
    for maxEvdir in os.listdir(path):
        res.append({"crossover-rate":["pc" , np.zeros(10, dtype=int)], 
    "cross-selector":["SelectC", np.zeros(7, dtype=int)],
    "crossover":["Crossover" , np.zeros(10, dtype=int)],
    "mutation-rate":["pm",np.zeros(10, dtype=int)], 
    "mut-selector":["SelectM",np.zeros(10, dtype=int)],
    "mutation":["Mutation", np.zeros(11, dtype=int)],
    "replacement":["Replacement" , np.zeros(11, dtype=int)],
    "pop-size":["pop-size", np.zeros(10, dtype=int)], 
    "offspring-size":["offspring-size" , np.zeros(10, dtype=int)]})
        for algodir in os.listdir(os.path.join(path,maxEvdir)): #fastgadir : directory of 50 runs of an elite configuration
            algo=algodir.split("_")
            for param in algo:
                name,val=param.split("=")[0],float(param.split("=")[1])
                if(name in {"pop-size" ,"offspring-size"}):
                    if(val%5==0):
                        res[-1][name][1][int(val//5) -1]+=1
                    else:
                        #print(res[-1][name][1],val//5)
                        res[-1][name][1][int(val//5)]+=1
                        
                elif(name in {"crossover-rate","mutation-rate"} ):
                    if(int(val*10)==10): #case of val=1
                        res[-1][name][1][-1]+=1
                    else :
                        #print(int(float(val)*10), name,pb,val)
                        res[-1][name][1][int(val*10)]+=1
                else :
                    res[-1][name][1][int(val)]+=1


    ind=0
    for maxEvdir in os.listdir(path):
        name="distribution_random_"+maxEvdir.split("_")[0]+".csv" #the end of the path must be /
        with open(os.path.join(distdir,name),"w+") as csvfile:
            csvfile.write("Op index, "+",".join(map(str,range(0,11)))+"\n")
        with open(os.path.join(distdir,name),"a") as csvfile:
            for param_name in res[ind].keys():
                #print(map(str,res[ind]),res[ind], ",".join(map(str,res[ind])))
                csvfile.write(res[ind][param_name][0]+","+ ",".join(map(str,res[ind][param_name][1]))+",-"*(11-len(res[ind][param_name][1])) +"\n")
                #print(str(i)+",",",".join(map(str,np.mean(aucs[i],1))))
        ind+=1
    #all problems
    name ="distribution_all_random_"+path.split("/")[-1]+".csv"
    with open(os.path.join(distdir,name),'w+') as csvfile:
        csvfile.write("Op index, "+",".join(map(str,range(0,11)))+"\n")

    with open(os.path.join(distdir,name),'a') as csvfile:
        for param_name in res[0].keys():
            #print(map(str,res[ind]),res[ind], ",".join(map(str,res[ind])))
            csvfile.write(res[0][param_name][0]+","+ ",".join(map(str,np.sum([res[i][param_name][1] for i in range(ind-1)],0)))+",-"*(11-len(res[0][param_name][1])) +"\n") #res[0] only for getting the name of parameters
            #print(str(i)+",",",".join(map(str,np.mean(aucs[i],1)))) 