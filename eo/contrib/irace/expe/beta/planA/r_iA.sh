#!/bin/bash
#run once each problem

echo "-------------------------Start the JOB : $(date --iso-8601=seconds)"
. /etc/profile.d/modules.sh
export MODULEPATH=${MODULEPATH}${MODULEPATH:+:}/opt/dev/Modules/Anaconda:/opt/dev/Modules/Compilers:/opt/dev/Modules/Frameworks:/opt/dev/Modules/Libraries:/opt/dev/Modules/Tools:/opt/dev/Modules/IDEs:/opt/dev/Modules/MPI
module load LLVM/clang-llvm-10.0
module load R

dir=$1
run=$2
budget_irace=$3
buckets=$4
myhome=$5
cp -r ${myhome}/R .
cp -r ${myhome}/irace_files_pA .
#cp -r /scratchbeta/zhenga/irace_files .
#chmod u+x ./fastga
outdir="${run}_$(date --iso-8601=seconds)_results_irace"
rundir=${dir}/${outdir}
mkdir -p ${rundir}
# Fore some reason, irace absolutely need those files...
cp ${myhome}/code/paradiseo/eo/contrib/irace/release/fastga  ${rundir}
cat ./irace_files_pA/example.scen | sed "s%\".%\"${rundir}%g" | sed "s/maxExperiments = 0/maxExperiments=${budget_irace}/" > ${rundir}/example.scen
cp ./irace_files_pA/default.instances ${rundir}
cp ./irace_files_pA/fastga.param ${rundir}
cp ./irace_files_pA/forbidden.txt ${rundir}
cat ./irace_files_pA/target-runner | sed "s/buckets=0/buckets=${buckets}/" > ${rundir}/target-runner
chmod u+x ${rundir}/target-runner

echo "---start $(date)"
time -p ./R/x86_64-pc-linux-gnu-library/3.6/irace/bin/irace --scenario ${rundir}/example.scen  > ${rundir}/irace.log 
echo "---end $(date)"
echo "End the JOB : $(date --iso-8601=seconds)------------------------------"
