#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

#get the auc average values of one experiment
figdir=sys.argv[1] # directory of a result of one experiment
#eg : ./fastga_results_all/fastga_results_planO/planO_maxExp\=100000_maxEv\=5n_2021-08-13T19\:04+02\:00_results_elites_all/raw

if("fastga_results_plan" in figdir):
    print("FID,",",".join(map(str,range(1,16))))
    aucs=[[] for i in range(19)]
    for fastgadir in os.listdir(os.path.join(figdir,"raw/data")): #fastgadir : directory of 50 runs of an elite configuration
        #cum=np.cumsum([0.1]*10)
        average=[]
        for fname in os.listdir(os.path.join(figdir,"raw/data",fastgadir)):
            with open(os.path.join(figdir,"raw/data",fastgadir,fname)) as fd:
                auc = float(fd.readlines()[0]) * -1
            average.append(auc)
        aucs[int(fastgadir.split("_")[0].split("=")[1])].append(average)
    #print(np.shape(aucs))



    for i in range(19):
        print(str(i)+",",",".join(map(str,np.mean(aucs[i],1))))

        
        
        



