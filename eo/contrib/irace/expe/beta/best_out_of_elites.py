#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
#cmd eg : 
#python3 best_out_of_elites.py ./fastga_results_all/fastga_results_planO/planO_maxExp=50000_maxEv=5n_2021-08-13T19:16+02:00_results_elites_all
#python3 best_out_of_elites.py ./fastga_results_all/fastga_results_random/maxEv=10000_nbAlgo=15_2021-08-21T20:53+02:00_results_randoms


#get the configuration of the best out of the elite
# recommendation suggested by 15 independant runs of irace

figdir=sys.argv[1] # directory of a result of one experiment
#eg : ./fastga_results_all/fastga_results_plan1/plan1_maxExp\=100000_maxEv\=5n_2021-08-13T19\:04+02\:00_results_elites_all/
#print(figdir.split('/')[-2], figdir.split('/'))
if("plan" in figdir.split('/')[-2]):
    print("Operator,","op. ,",",".join(map(str,range(1,20))))

    column={"pc" : 101, "SelectC": 7, "Crossover" : 10, "pm": 101,"SelectM" : 7, "Mutation": 11, "Replacement" : 11, "pop-size": 50, "offspring-size" : 50}
    nbparam=(len(os.listdir(os.path.join(figdir,"raw/data"))[0].split("_"))-1) #-1 car il y a le pb

    if( nbparam<len(column)):
        del column["pop-size"]
        del column["offspring-size"]
    configs=[(-1,-1)]*19 #tuple(auc,config)
    res=np.zeros((nbparam,19))
    for fastgadir in os.listdir(os.path.join(figdir,"raw/data")): #fastgadir : directory of 50 runs of an elite configuration
        #cum=np.cumsum([0.1]*10)
        average=[]
        for fname in os.listdir(os.path.join(figdir,"raw/data",fastgadir)):
            with open(os.path.join(figdir,"raw/data",fastgadir,fname)) as fd:
                auc = float(fd.readlines()[0]) * -1
            average.append(auc)
        pb=int(fastgadir.split("_")[0].split("=")[1])
        new_auc=np.mean(average)
        if(configs[pb][0]<new_auc):
            configs[pb]=(new_auc,fastgadir)


    for pb in range(19):
        config=configs[pb][1].split("_")
        configparam=[p.split("=")[1] for p in config[1:]]
        res[:,pb]=configparam

    ind=0 #index of param_name
    for param_name in column.keys():
        #print(map(str,res[ind]),res[ind], ",".join(map(str,res[ind])))
        print(param_name+","+str(column[param_name])+",", ",".join(map(str,res[ind])))
        ind+=1
        #print(str(i)+",",",".join(map(str,np.mean(aucs[i],1))))

if("maxEv" in figdir.split('/')[-2]):
    print("Operator,","op. ,",",".join(map(str,range(1,20))))

    column={"pc" : 101, "SelectC": 7, "Crossover" : 10, "pm": 101,"SelectM" : 7, "Mutation": 11, "Replacement" : 11, "pop-size": 50, "offspring-size" : 50}
    nbparam=(len(os.listdir(figdir)[0].split("_")))
    if( nbparam<len(column)):
        del column["pop-size"]
        del column["offspring-size"]
    configs=[(-1,-1)]*19 #tuple(auc,config)
    bests=np.zeros((nbparam,19))
    
    for algodir in os.listdir(figdir): #algodir : directory of one random algo
        for fname in os.listdir(os.path.join(figdir,algodir,"data")): #fname : directory of 50 runs of fastga for one pb
            average=[]
            for res in os.listdir(os.path.join(figdir,algodir,"data",fname)):
                with open(os.path.join(figdir,algodir,"data",fname,res)) as fd:
                    auc = float(fd.readlines()[0]) * -1
                average.append(auc)
            pb=int(fname.split("_")[0].split("=")[1])
            new_auc=np.mean(average)
            if(configs[pb][0]<new_auc):
                configs[pb]=(new_auc,algodir)


    for pb in range(19):
        config=configs[pb][1].split("_")
        configparam=[p.split("=")[1] for p in config]
        bests[:,pb]=configparam

    ind=0 #index of param_name
    for param_name in column.keys():
        #print(map(str,res[ind]),res[ind], ",".join(map(str,res[ind])))
        print(param_name+","+str(column[param_name])+",", ",".join(map(str,bests[ind])))
        ind+=1
