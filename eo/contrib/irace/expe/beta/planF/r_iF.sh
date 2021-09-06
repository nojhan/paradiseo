#!/bin/bash
#run once each problem
dir=$1
run=$2
budget_irace=$3
buckets=$4
myhome=$5

echo "---------------start JOB ${run} $(date --iso-8601=seconds)"
. /etc/profile.d/modules.sh
export MODULEPATH=${MODULEPATH}${MODULEPATH:+:}/opt/dev/Modules/Anaconda:/opt/dev/Modules/Compilers:/opt/dev/Modules/Frameworks:/opt/dev/Modules/Libraries:/opt/dev/Modules/Tools:/opt/dev/Modules/IDEs:/opt/dev/Modules/MPI
module load LLVM/clang-llvm-10.0
module load R

cp -r ${myhome}/R .
cp -r ${myhome}/irace_files_pF .
#cp -r /scratchbeta/zhenga/irace_files .
#chmod u+x ./fastga
outdir="${run}_$(date --iso-8601=seconds)_results_irace"
for pb in $(seq 0 18) ; do
    echo "Problem ${pb}... "
    res="results_problem_${pb}"
    mkdir -p ${dir}/${outdir}/${res}
    # Fore some reason, irace absolutely need those files...
    cp ${myhome}/code/paradiseo/eo/contrib/irace/release/fastga  ${dir}/${outdir}/${res}
    cat ./irace_files_pF/example.scen | sed "s%\".%\"${dir}/${outdir}/${res}%g" |  sed "s/maxExperiments = 0/maxExperiments=${budget_irace}/" > ${dir}/${outdir}/${res}/example.scen
    cp ./irace_files_pF/default.instances ${dir}/${outdir}/${res}
    cp ./irace_files_pF/fastga.param ${dir}/${outdir}/${res}
    cp ./irace_files_pF/forbidden.txt ${dir}/${outdir}/${res}
    cat ./irace_files_pF/target-runner | sed "s/--problem=0/--problem=${p}/" |  sed "s/buckets=0/buckets=${buckets}/" > ${dir}/${outdir}/${res}/target-runner
    chmod u+x ${dir}/${outdir}/${res}/target-runner

    echo "---start $(date)"
    time -p ./R/x86_64-pc-linux-gnu-library/3.6/irace/bin/irace --scenario ${dir}/${outdir}/${res}/example.scen  > ${dir}/${outdir}/${res}/irace.log 
    echo "---end $(date)"
done
echo "end JOB ${run} $(date --iso-8601=seconds)---------------"
