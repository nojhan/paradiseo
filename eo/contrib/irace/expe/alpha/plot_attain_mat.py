#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

for p in range(18):

    print("Pb",p,end=": ")
    datadir="attain_mat_{pb}".format(pb=p)

    try:
        os.mkdir(datadir)
    except FileExistsError:
        pass

    for i in range(50):
        cmd="./release/fastga --seed={i} \
            --crossover-rate=2 --cross-selector=2 --crossover=5 --mutation-rate=2 --mut-selector=2 --mutation=1 --replacement=0 \
            --problem={pb} --buckets=20 --output-mat 2>/dev/null > {dir}/output_mat_{i}.csv"\
            .format(dir=datadir, i=i, pb=p)
        # print(cmd)
        print(i,end=" ",flush=True)
        os.system(cmd)


    matrices=[]
    for root, dirs, files in os.walk(datadir):
        for filename in files:
            matrices.append( np.genfromtxt(datadir+"/"+filename,delimiter=',') )

    agg = matrices[0]
    for mat in matrices[1:]:
        agg += mat

    # print(agg)

    plt.rcParams["figure.figsize"] = (3,3)
    plt.gca().pcolor(agg, edgecolors='grey', cmap="Blues")
    plt.gca().set_xlabel("Time budget")
    plt.gca().set_ylabel("Target")
    plt.savefig("aittain_map_{pb}.png".format(pb=p), bbox_inches='tight')

    print(".")
