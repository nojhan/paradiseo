#!/bin/bash
#run once each problem

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
cp -r ${myhome}/irace_files_pO .

outdir="${run}_$(date --iso-8601=seconds)_results_irace"
echo "start a job  $(date -Iseconds)"

for pb in $(seq 0 18) ; do
    echo "Problem ${pb}... "
    res="results_problem_${pb}"
    mkdir -p ${dir}/${outdir}/${res}
    # Fore some reason, irace absolutely need those files...
    cp ${myhome}/code/paradiseo/eo/contrib/irace/release/fastga ${dir}/${outdir}/${res}

    cat ./irace_files_pO/example.scen | sed "s%\".%\"${dir}/${outdir}/${res}%g" |  sed "s/maxExperiments = 0/maxExperiments=${budget_irace}/" > ${dir}/${outdir}/${res}/example.scen
    cp ./irace_files_pO/default.instances ${dir}/${outdir}/${res}
    cp ./irace_files_pO/fastga.param ${dir}/${outdir}/${res}
    cat ./irace_files_pO/target-runner | sed "s/--problem=0/--problem=${pb}/" > ${dir}/${outdir}/${res}/target-runner
    chmod u+x ${dir}/${outdir}/${res}/target-runner

    echo "---start $(date)"
    time -p ./R/x86_64-pc-linux-gnu-library/3.6/irace/bin/irace --scenario ${dir}/${outdir}/${res}/example.scen  > ${dir}/${outdir}/${res}/irace.log  
    echo "---end $(date)"
    
    echo "done run : ${run} pb : ${pb}"
    date -Iseconds
done

echo "end a job $(date -Iseconds)---------------------"

