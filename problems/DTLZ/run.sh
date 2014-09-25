#population size
POP_SIZE=100

#ZDT number (1, 2, 3, 4 or 6)
EVAL=1

#vector Size
VEC_SIZE=30

#number of objectives
NB_OBJ=3

#PROBABIITY FOR SBXCROSSOVER
P_CROSS=1.0

#EXTERNAL PROBABILITY FOR POLYNOMIAL MUTATION
EXT_P_MUT=1.0

#INTERNAL PROBABILITY FOR POLYNOMIAL MUTATION
INT_P_MUT=0.083333

#ARCHIVE SIZE (ONLY FOR SPEA2)
ARC_SIZE=100

#K-TH DISTANCE (ONLY FOR SPEA2)
K=10

#number of evaluation
NB_EVAL=500

#Time 
TIME=0

#seed
SEED=1

DTLZ4=100

SPEA2="SPEA2.out"

IBEA="IBEA.out"

NSGA="NSGAII.out"

./build/application/DTLZ_SPEA2 --eval=$i --dtlz4_param=$DTLZ4 --vecSize=$VEC_SIZE --nbObj=$NB_OBJ --pCross=$P_CROSS --extPMut=$EXT_P_MUT --intPMut=$INT_P_MUT --arcSize=$ARC_SIZE --nbEval=$NB_EVAL --time=$TIME --k=$K -o=$SPEA2

./build/application/DTLZ_IBEA --eval=$i --dtlz4_param=$DTLZ4 --vecSize=$VEC_SIZE --nbObj=$NB_OBJ --pCross=$P_CROSS --extPMut=$EXT_P_MUT --intPMut=$INT_P_MUT --nbEval=$NB_EVAL --time=$TIME -o=$IBEA

./build/application/DTLZ_NSGAII --eval=$i --dtlz4_param=$DTLZ4 --vecSize=$VEC_SIZE --nbObj=$NB_OBJ --pCross=$P_CROSS --extPMut=$EXT_P_MUT --intPMut=$INT_P_MUT --nbEval=$NB_EVAL --time=$TIME -o=$NSGA

