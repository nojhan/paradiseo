#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

#get the distribution of operators variants recommended by 15 runs of irace for all problems and each problem
#also get an csv file
#pc and pm 10 possibilities :   [0-0.1[  [0.1-0.2[ [0.2-0.3[  [0.3-0.4[  [0-0.5[  [0.5-0.6[ ...[0.9-1[
#pop-size and offspring-size 10 possibilities : 0-5 5-10, 10-15 15-20  20-25  25-30 30-35- 35-40 40-45 45-50

path=sys.argv[1] # directory of a result of one experiment
#eg : ./fastga_results_all/fastga_results_planO/planO_maxExp\=100000_maxEv\=5n_2021-08-13T19\:04+02\:00_results_elites_all/

if("fastga_results_plan" in path):
        #column : [operator : nbpossibilities]
    distdir=sys.argv[2]
    try:
        os.makedirs(distdir)
    except FileExistsError:
        pass

    nbparam=(len(os.listdir(os.path.join(path,"raw/data"))[0].split("_"))-1)

    if( nbparam==7):
        res=[{"crossover-rate":["pc" , np.zeros(10, dtype=int)], 
        "cross-selector":["SelectC", np.zeros(7, dtype=int)],
        "crossover":["Crossover" , np.zeros(10, dtype=int)],
        "mutation-rate":["pm",np.zeros(10, dtype=int)], 
        "mut-selector":["SelectM",np.zeros(7, dtype=int)],
        "mutation":["Mutation", np.zeros(11, dtype=int)],
        "replacement":["Replacement" ,np.zeros(11, dtype=int)]} for i in range(19)]
    else:
        res=[{"crossover-rate":["pc" , np.zeros(10, dtype=int)], 
        "cross-selector":["SelectC", np.zeros(7, dtype=int)],
        "crossover":["Crossover" , np.zeros(10, dtype=int)],
        "mutation-rate":["pm",np.zeros(10, dtype=int)], 
        "mut-selector":["SelectM",np.zeros(7, dtype=int)],
        "mutation":["Mutation", np.zeros(11, dtype=int)],
        "replacement":["Replacement" , np.zeros(11, dtype=int)],
        "pop-size":["pop-size", np.zeros(10, dtype=int)], 
        "offspring-size":["offspring-size" , np.zeros(10, dtype=int)]} for i in range(19)]
        

    for fastgadir in os.listdir(os.path.join(path,"raw/data")): #fastgadir : directory of 50 runs of an elite configuration
        algo=fastgadir.split("_")
        pb=int(fastgadir.split("_")[0].split("=")[1])
        for param in algo[1:]:
            name,val=param.split("=")[0],float(param.split("=")[1])
            if(name in {"pop-size" ,"offspring-size"}):
                if(val%5==0):
                    res[pb][name][1][int(val//5) -1]+=1
                else:
                    #print(res[pb][name][1],val//5)
                    res[pb][name][1][int(val//5)]+=1
                    
            elif(name in {"crossover-rate","mutation-rate"} ):
                if(int(val*10)==10): #case of val=1
                    res[pb][name][1][-1]+=1
                else :
                    #print(int(float(val)*10), name,pb,val)
                    res[pb][name][1][int(val*10)]+=1
            else :
                res[pb][name][1][int(val)]+=1



    for pb in range(19):
        name="distribution_pb="+str(pb)+"_"+path.split("/")[-2]+".csv" #the end of the path must be /
        with open(os.path.join(distdir,name),"w+") as csvfile:
            csvfile.write("Op index, "+",".join(map(str,range(0,11)))+"\n")
        with open(os.path.join(distdir,name),"a") as csvfile:
            for param_name in res[pb].keys():
                #print(map(str,res[ind]),res[ind], ",".join(map(str,res[ind])))
                csvfile.write(res[pb][param_name][0]+","+ ",".join(map(str,res[pb][param_name][1]))+",-"*(11-len(res[pb][param_name][1])) +"\n")
                #print(str(i)+",",",".join(map(str,np.mean(aucs[i],1))))

    #all problems
    name ="distribution_all_pb_"+path.split("/")[-1]+".csv"
    with open(os.path.join(path,"raw",name),'w+') as csvfile:
        csvfile.write("Op index, "+",".join(map(str,range(0,11)))+"\n")

    with open(os.path.join(path,"raw",name),'a') as csvfile:
        for param_name in res[0].keys():
            #print(map(str,res[ind]),res[ind], ",".join(map(str,res[ind])))
            csvfile.write(res[0][param_name][0]+","+ ",".join(map(str,np.sum([res[i][param_name][1] for i in range(19)],0)))+",-"*(11-len(res[0][param_name][1])) +"\n") #res[0] only for getting the name of parameters
            #print(str(i)+",",",".join(map(str,np.mean(aucs[i],1)))) 