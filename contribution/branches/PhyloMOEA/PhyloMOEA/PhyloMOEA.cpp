#include <eo>
#include <moeo>
#include <peo>
#include <PhyloMOEO.h>
#include <PhyloMOEO_operators.h>
#include <PhyloMOEO_init.h>
#include <PhyloMOEO_eval.h>
#include <PhyloMOEO_archive.h>
#include <PhyloMOEOProbMatrixContainerUpdater.h>
#include <PhyloMOEO_packunpack.h>
#include <moeoNSGAII2.h>
#include <utils/moeoBestObjVecStat.h>
#include <utils/moeoAverageObjVecStat.h>
#include <PhyloMOEOPartitionStat.h>
#include <eoCountedFileMonitor.h>
#include <eoSingleFileCountedStateSaver.h>
#include <vectorSortIndex.h>

#include <utils.h>
#include <apply.h>


gsl_rng *rn2;
RandomNr *rn;
//Sequences *seq;
long seed;
//vector<phylotreeIND> arbores;
string datafile,usertree, expid, path, algotype;
double pcrossover, pmutation, kappa, alpha;
unsigned int ngenerations,  popsize, ncats;
ofstream exp_data,evolution_data, best_media_scores, final_trees, final_pareto_trees, clades_pareto, clades_final,final_scores,pareto_scores;
LikelihoodCalculator *lik_calc_ptr;
phylotreeIND *templatetree_ptr;
ProbMatrixContainer *probmatrixs_ptr;

int main(int argc, char *argv[])
{
	// measures execution time
	struct timeval tempo1, tempo2, result;
	gettimeofday(&tempo1, NULL);
	peo :: init( argc, argv );




	eoParser parser(argc, argv);

	seed = parser.createParam((unsigned int)(time(NULL)), "seed", "Random Seed", 's',"Param").value();
	
	popsize = parser.createParam((unsigned int)(50), "popSize", "Population size", 'n',"Param").value();
	pcrossover = parser.createParam((double)(0.8), "pcross", "Crossover Rate", 'c',"Param").value();
	pmutation = parser.createParam((double)(0.1), "pmut", "Mutation Rate", 'm',"Param").value();
	ngenerations = parser.createParam((unsigned int)(500), "nGen", "Number of Generations", 'g',"Param").value();
	kappa = parser.createParam((double)(4.0), "kappa", "Kappa value", 'k',"Param").value();
	alpha = parser.createParam((double)(2.0), "alpha", "Alpha value", 'a',"Param").value();
	ncats = parser.createParam((unsigned int)(4), "nCat", "Number of Categories", 'r',"Param").value();
	datafile = parser.createParam(string(), "data", "Datafile", 'd',"Param").value();
	usertree = parser.createParam(string(), "treef", "Treefile", 't',"Param").value();
	path = parser.createParam(string(), "path", "Treefile", 'p',"Param").value();
	algotype = parser.createParam(string("nsgaii"), "algo", "Algorith, Type", 'b',"Param").value();
 	ostringstream convert;
 	convert << seed;
	expid = parser.createParam(convert.str(), "expid", "Experiment ID", 'e',"Param").value();

	if(getNodeRank()==1) welcome_message();
	if( datafile.size()==0 )
	{
		
	  	if(getNodeRank()==1) parser.printHelp( cout );
		return(-1);
	}
	

	// all nodes read datafile and prepares likelihood and parsimony calculation
	string filename = path + datafile;
	cout << "\n\nReading Sequence Datafile...";
	Sequences seq(filename.c_str());
	cout << " done.\n";
	// calculate datafile
	cout << "calculating pattersn..." << getNodeRank() << endl;
	seq.calculate_patterns();
	seq.calculate_frequences();

 	rn2 = gsl_rng_alloc(gsl_rng_default);
 	rn = new RandomNr(seed);
	phylotreeIND templatetree( rn, seq, rn2);
	templatetree_ptr = &templatetree;
	ParsimonyCalculator parsi_calc(templatetree);
 	SubstModel modelHKY( seq, SubstModel::HKY85);
 	modelHKY.init();
 	modelHKY.set_kappa(kappa); 
 	ProbMatrixContainer probmatrixs(modelHKY);
	probmatrixs_ptr = &probmatrixs;
	LikelihoodCalculator lik_calc(templatetree, modelHKY, probmatrixs,ncats);
	lik_calc.set_alpha(alpha);
 	modelHKY.init();

	PhyloEval byobj( parsi_calc, lik_calc );
	Phyloraninit initializer(templatetree);
	eoPop<PhyloMOEO> population(popsize, initializer);

	peoMoeoPopEval <PhyloMOEO> eval(byobj);


	// Only the master node read the initial trees and writes ouput files
	if(getNodeRank()==1)
	{

		cout << "\n\nReading Initial Trees...";
		if( usertree.size() >0)	
		{
			filename = path + usertree;
			readtrees(filename.c_str(), population);
		}
		cout << " done.\n";

		cout << "\n\nCreating output files...";
	
		try{
			filename = path + datafile + "_exp_param_" + expid  + ".txt";
			exp_data.open(filename.c_str());
			exp_data.precision(15);
			exp_data.setf(ios::fixed);
	
	
	
			if(  !exp_data.is_open()	)
			{
				throw( ExceptionManager(12) );
			}
			cout << " done.\n";
		}
		catch ( ExceptionManager e )
		{
			e.Report();
		}
				// create the moea	<	
		save_exp_params(exp_data);
		seq.save_seq_data(exp_data); 

	}

	eoGenContinue<PhyloMOEO> continuator(ngenerations);
	eoCheckPoint<PhyloMOEO> cp(continuator);
	Phylomutate mutator;
	Phylocross crossover;
	eoSequentialOp<PhyloMOEO> operadores;
	operadores.add(crossover,pcrossover);
	operadores.add(mutator,pmutation);
	PhyloMOEOProbMatrixContainerUpdater probmatrixupdater(probmatrixs);
	cp.add( probmatrixupdater );

	moeoAverageObjVecStat <PhyloMOEO> *bestfit;
	moeoBestObjVecStat <PhyloMOEO> *avgfit;
	eoPopStat<PhyloMOEO> *popstats;
	
	eoCountedFileMonitor *media_scores, *evolution_scores;
	eoStdoutMonitor *monitor;
	eoValueParam<unsigned> *generationCounter;
	eoIncrementor<unsigned> *increment;


	// only the master output files
	if(getNodeRank()==1){
			
			generationCounter = new eoValueParam<unsigned> (0, "Gen.");
			increment = new eoIncrementor<unsigned>(generationCounter->value()); 
			cp.add(*increment); 

			monitor = new eoStdoutMonitor(false);
			monitor->add(*generationCounter);

			bestfit = new moeoAverageObjVecStat <PhyloMOEO>();
			avgfit = new moeoBestObjVecStat <PhyloMOEO>();
			popstats = new eoPopStat<PhyloMOEO> ();

			media_scores = new eoCountedFileMonitor( 2, path + datafile + "_media_scores_" + expid + ".txt", "\t", true,true );
			media_scores->add( *avgfit) ;
			media_scores->add( *bestfit );

			evolution_scores = new eoCountedFileMonitor( 2, path + datafile + "_evolution_data_" + expid + ".txt", "\n", true,true );
			evolution_scores->add( *popstats);
			cp.add( *media_scores);
			cp.add( *evolution_scores );
			cp.add( *popstats);
			cp.add( *bestfit);
			cp.add( *avgfit);
			cp.add( *monitor); 
	}
	
	
	if(algotype == "ibea")
	{
		moeoAdditiveEpsilonBinaryMetric < ObjectiveVector > metric;
		moeoIBEA < PhyloMOEO > algo (cp, eval, operadores, metric);
		if(getNodeRank()==1){
		cout << "\n\nRunning IBEA ..." << endl;	}
  		peoWrapper parallelEA( algo, population);
  		eval.setOwner(parallelEA);
		peo :: run();
		peo :: finalize();
		//algo(population);
	}	
	else
	{
		moeoNSGAII < PhyloMOEO > algo (cp, eval, operadores);
		if(getNodeRank()==1){
		cout << "\n\nRunning NSGA-II ..." << endl;	}
  		peoWrapper parallelEA( algo, population); 
  		eval.setOwner(parallelEA);
		peo :: run();
		peo :: finalize();
		//algo(population);
	}

	if (getNodeRank()==1) 
	{
		delete media_scores;
		delete evolution_scores;
		delete bestfit;
		delete avgfit;
		delete popstats;
		delete monitor;
		delete generationCounter;
		delete increment;
	

		cout << "\nCalculating Final Solutions...";
		cout << "  done\n";
	
	}

	PhyloMOEOFinalSolutionsArchive finalsolutions;
	if(getNodeRank()==1)finalsolutions.operator()(population);
	//cout << "en el nodo " << getNodeRank() << " popsize " << population.size() << endl;
	//cout << "en el nodo " << getNodeRank() << " archsize " << finalsolutions.size() << endl;
	//finalsolutions[0].get_tree().printNewick(cout);
	lik_calc_ptr = &lik_calc;
	// make the optimization phase also in parallel
	peo :: init (argc, argv);
	peoMultiStart <PhyloMOEO> ParallelLKOptimizationInit (optimize_solution);
	peoWrapper ParallelLKOptimization (ParallelLKOptimizationInit, finalsolutions);
	ParallelLKOptimizationInit.setOwner(ParallelLKOptimization);
	if (getNodeRank()==1) cout << "\nOptimizing tree branch lenghts...\n";
	peo :: run( );
	peo :: finalize( );
	
	if (getNodeRank()==1)
	{
	
		//remove_final_solutions( population );
		// optimize remaining solutions

		
		//optimize_solutions( finalsolutions, lik_calc );
		cout << "\nReevaluating individuals \n";
		apply<PhyloMOEO> ( byobj, finalsolutions );
	
		finalsolutions.save_scores(path + datafile + "_final_scores_" + expid + ".txt","#Final Solutions Scores");
		finalsolutions.save_trees(path + datafile + "_final_trees_" + expid + ".txt");
		cout << "\ndone \n";
		
		// print the optimized solutions
		//print_scores_pop( -2, population, evolution_data);
	
		//print_scores_pop( -2, population, final_scores);
	
		//save_trees(finalsolutions, final_trees);
		cout << "\n\nCalculating Final Solutions clade support...";
		
		PhyloMOEOPartitionStat splitstats;
		splitstats(finalsolutions);
		eoFileMonitor finalsplitstatsaver(path+datafile+"_clades_final_"+expid+".txt");
		finalsplitstatsaver.add(splitstats);
		finalsplitstatsaver(); 
		//cout << splitstats.value() << endl;
		//partition_map split_frequences;
		//calculate_frequence_splits(finalsolutions,split_frequences);
		cout << " done\n";
		//save_partitions(splitstats.value(), clades_final);
		//split_frequences.clear();
		// remove dominate solutions
		cout << "\nCalculating Pareto-optimal Solutions...";
	
		PhyloMOEOParetoSolutionsArchive paretosolutions;
		paretosolutions.operator()(finalsolutions);
		paretosolutions.save_scores(path + datafile + "_pareto_scores_" + expid + ".txt","#Pareto Solutions Scores");
		paretosolutions.save_trees(path + datafile + "_pareto_trees_" + expid + ".txt");
		cout << " done\n";
		// print final pareto trees
		//save_trees( paretosolutions, final_pareto_trees);
		cout << "\nCalculating Pareto-optimal Solutions clade support...";
		splitstats(paretosolutions);
		//calculate_frequence_splits(paretosolutions,split_frequences);
		eoFileMonitor paretosplitstatsaver(path+datafile+"_clades_pareto_"+expid+".txt");
		paretosplitstatsaver.add(splitstats);
		paretosplitstatsaver();
	
	//	save_partitions(splitstats.value(), clades_pareto);
		//split_frequences.clear();
		cout << " done\n";
		exp_data.close();
		evolution_data.close();
		pareto_scores.close();
		final_scores.close();
		best_media_scores.close();
		final_trees.close();
		final_pareto_trees.close();
		clades_pareto.close();
		clades_final.close();
		cout << "\nPhyloMOEA execution finishes !\n";
		gettimeofday(&tempo2, NULL);
		timeval_subtract(&result,&tempo2,&tempo1);	

		long remainder = result.tv_sec % 3600;
		long hours = (result.tv_sec - remainder)/3600;
		long seconds = remainder % 60;
		long minutes = (remainder - seconds) / 60;
		cout << "Execution time :  ";
		cout.width(3);
		cout.fill(' ');
		cout << hours << ":";
		cout.width(2);
		cout.fill('0');
		cout << minutes << ":";
		cout.width(2);
		cout.fill('0');
		cout << seconds << "." << result.tv_usec << "(" << result.tv_sec << ")" << endl;
	}
	gsl_rng_free(rn2);
	//	delete probmatrixs;
	delete rn;
	return 0;
} 


