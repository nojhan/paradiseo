#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas

#eg : python3 ./rep_std_mean_selected.py ./hist_and_csv/distribution_op_fastga_results_planF
#get the std of the selected problem
path=sys.argv[1] # directory of each distribution by pb
lpb={13,14,15,16,18} #set of pb selected
#column : [operator : nbpossibilities]
distdir=path+"/rep_std_mean"
try:
    os.makedirs(distdir)
except FileExistsError:
    pass

res=[]
for csvfile in os.listdir(os.path.join(path)):
    if(int(csvfile.split("_")[1].split("=")[1]) in lpb):
        print(csvfile)
        res.append(pandas.read_csv(os.path.join(path,csvfile)))

#assert(len(res[0])==len(res[1]) , "each csv file does not have the same line " #check if the number of param is eq in each csv file


name ="std_rep_pb={}".format(str(lpb))+"".join(map(str,path.split("/")[-3].split("_")[:3]))+".csv"
with open(os.path.join(distdir,name),'w+') as fd:
    fd.write("Op index, "+",".join(map(str,range(0,11)))+"\n")
globalname="rep_all_pb={}".format(str(lpb))+"".join(map(str,path.split("/")[-3].split("_")[:3]))+".csv"
with open(os.path.join(distdir,globalname),'w+') as fd:
    fd.write("Op index, "+",".join(map(str,range(0,11)))+"\n")
meanname="mean_rep_pb={}".format(str(lpb))+"".join(map(str,path.split("/")[-3].split("_")[:3]))+".csv"
with open(os.path.join(distdir,meanname),'w+') as fd:
    fd.write("Op index, "+",".join(map(str,range(0,11)))+"\n")
#print(res)
limparam=[10,7,10,10,7,11,11,10,10]
for i in range(1,10): #9 nb parameters
    npval=np.zeros((len(res),limparam[i-1]),dtype=int)
    for pb in range(len(res)): 
        print(i,np.array(np.array(res[pb][i-1:i])[0]),np.array(np.array(res[pb][i-1:i])[0][1:limparam[i-1]+1]))
        npval[pb,:]=np.array(np.array(res[pb][i-1:i])[0][1:limparam[i-1]+1],dtype=int)
    nameparam=np.array(res[pb][i-1:i])[0][0]
    line= ",".join(map(str,np.std(npval,0)))+",-"*(11-limparam[i-1])
    print("ligne ",line)

    with open(os.path.join(distdir,name),'a') as fd:
        fd.write(nameparam+","+line+"\n")
    line= ",".join(map(str,np.sum(npval,0)))+",-"*(11-limparam[i-1])
    with open(os.path.join(distdir,globalname),'a') as fd:
        fd.write(nameparam+","+line+"\n")
    line= ",".join(map(str,np.mean(npval,0)))+",-"*(11-limparam[i-1])
    with open(os.path.join(distdir,meanname),'a') as fd:
        fd.write(nameparam+","+line+"\n")