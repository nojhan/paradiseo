############################################
#Explanation of the experimental plans and the validation runs

############################################
1. INTRODUCTION

The aim of all the scripts is to make the experimental plans for Algorithm Configuration for Genetic Algorithms by using a fully modular benchmarking pipeline design of this article https://arxiv.org/abs/2102.06435 . 

You can upload the data in : https://zenodo.org/record/5479538#.YTaT0Bnis2w 

Plan A is an experimental plan for finding an efficient algorithm for all the functions that we consider.

Plan F is an experimental plan for finding an efficient algorithm for each function that we consider.

Plan R is an experimental plan for getting random algorithms.

Plan O is the reproduction of the experimental plan of the article.

2. VOCABULARIES

* maxExp : means maximum Experiments,  the budget for irace
* maxEv : means maximum evaluation, the budget for FastGA algorithms

*dataFAR : directory which we store all the experiment data of Plan F and Plan A, created when you run run_exp.sh

* dataA, dataF 
dataA is a directory which we store all the runs of an experiment plan for several budgets
eg : /dataA/planA_maxExp=*_maxEv=**_$(data), * is a value of maxExp, and ** is a value of maxEv


*fastga_results_all : directory which we store all the data for validation runs. It constains only 3 subdirectories (fastga_results_planF, fastga_results_planA, fastga_results_planO, fastga_results_random), created by running run_exp.sh

* fastga_results_planF, fastga_results_planA, fastga_results_random, fastga_results_planO
Each directory store the data for validation runs of each experiment plan.
fastga_random directory are created by running run_exp.sh
fastga_results_planF, fastag_results_planO and fastag_results_planA are created only after you have data in the dataA or dataF or dataO directories.


* planA_*, planF_*, planO_*
If the planA_* or planF_* or planO_* are in the dataFAR directory, the directory contains the data of experimental plan. This means that each plan contains the result of 15 runs of irace stored in irace.log file, and the data are provided by run_exp.sh.

If the planA_* or planF_* or planO_* directories are in the fastga_results_planA or fastga_results_planF, these directories contain the data of 50 validation runs by running all the best algorithms of each plan stores in dataFAR. The data are provided by running run_res.sh


*fastag_all_results : contains the directories of the validation run data.

*fastga_results_planF, fastga_results_planA and fastga_results_random contain respectively the validation run data of Plan F, Plan A and Plan R.


3. DESCRIPTION

The directory which you load all the scripts contains : 

 * bash files :
	-run_res.sh : submit to the cluster all the experiment plan, get all the data we need for the plan F, plan A and plan R.
	-run_exp.sh : submit to the cluster for getting all the data for validation runs of each data A and data F provided by running run_res.sh
	-run_random.sh : script for getting random algorithms and the data for validation runs for each problem
        -testrandom.sh : change the budget fastga (maxEv) in this file if you need, by running this file, you submit plan R job in the cluster
	
        -csv_all_bests.sh : script for getting all the best configurations of each plan in a dataF or a dataA directories

        -run_elites_planA.sh : script for validation runs of plan A by giving a csv file of each best configuration. This file is provided by running parseA_irace_bests.py.
	-run_elites_planB.sh
	-fastga_elites_all.sh : run this file, by giving a directory csv_plan* of csv files ( must only contains the csv file of the same plan, eg : csv_planF) and a run_elites_plan*.sh (* is the name of the plan, eg run_elites_planF.sh), by running this file you get all the validation runs of each csv file. Each csv file contains the best configuration (you get these csv files by running csv_all_bests.sh) 

 * python files : 
	-parseA_irace_bests.py : for parsing the irace.log file of each data provided by running irace. By giving a bounch of directories of one experiment
	-parseF_irace_bests.py : for the plan plan F and plan O(in the plan O csv, there are label offspringsize and popsize, but there are not values)

 * 6 directories : 
	-irace_files_pA : 
		-default.instances
		-example.scen
		-fastga.param
		-forbidden.txt
		-target-runner

	-irace_files_pF :
		-default.instances : 
		-example.scen
		-fastga.param
		-forbidden.txt
		-target-runner

	-irace_files_pO :
		-default.instances : 
		-example.scen
		-fastga.param
		-target-runner

	-planA :
		-riaA.sh : for running 15 times r_iA.sh file by submitting to the mesu cluster
		-r_iA.sh : for running irace for all the problems

	-planF :
		-riaF.sh : for running 15 times r_iF.sh file by submitting to the mesu cluster
		-r_iF.sh : for running irace for each problem we considered
	-planO :
		-riaO.sh : for running 15 times r_iO.sh file by submitting to the mesu cluster
		-r_iO.sh : for running irace for each problem we considered


The directories planA, planF contain the scripts to run one experiment of Plan A and Plan F.

The directories irace_files_pA, irace_files_pO and irace_files_pF contain the scripts needing for calling irace for one experiment of Plan A, Plan O and Plan F. [Look at the irace package : User Guide for more information]


5. CONCLUSION

For getting all the experiment data and the validation run data run run_exp.sh file first, after run_exp.sh file finished to execute and there is all the data in the dataFAR (ie : in the cluster, all the jobs finished to execute) run_res.sh data.

Warning : run_exp.sh may take few days or few weeks depending on the Budget you ask, do not run_res.sh if in dataFAR there are data which are not finished to execute in the cluster, or jobs killed. Do not forget to remove directories of plan which are not complete.




############################################
#Scripts for getting histograms and csv files of validation runs results.

############################################

get histograms or csv files for random data :
-hist_join_random.py  : get one histogram for a plan by budget
-dist_op_random.py  : get csv files of the distribution of operators by problems 

get histograms or csv files for plan O,F,A :
-hist_join.py
-dist_op_all.py
-parse_auc_average  # get the mean auc value of each problem and each irace run

get histograms for plan F, A , R, O
-hist_by_pb_budget_plan.py   : get histograms by problem
-hist_by_FARO_pb.py  : 
-hist_by_FARO.py
-best_out_of_elites.py : get the best algorithm found among 15 runs of irace, for a plan


files to call all these files : 
-csv_all.sh : get all the csv files (average of auc, best out ..), call best_out_of_elites.py, parse_auc_average.py, dist_op_*.py
-hist_all.sh : get all the histograms, call each hist_*.py file

file for other goal :
-mwtestU.py ; csv file for selected problems which irace algorithms gave better performances than random algorithms
-rep_std_mean_selected.py : to get the std, mean and the distribution of operators of the selected problems 





############################################
#Summary

############################################

Get the experiment data :
run : bash run_exp.sh

-----------Only after you have the experiment data:
Get the validation run data :
run : bash run_res.sh

Get histograms :
run : bash hist_all.sh 

Get csv files of validation run data :
run : bash csv_all.sh

