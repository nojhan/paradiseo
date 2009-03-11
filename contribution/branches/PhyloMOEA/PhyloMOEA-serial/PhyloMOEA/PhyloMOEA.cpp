#include <eo>
#include <moeo>
#include <PhyloMOEO.h>
#include <PhyloMOEO_operators.h>
#include <PhyloMOEO_init.h>
#include <PhyloMOEO_eval.h>
#include <PhyloMOEO_archive.h>
#include <PhyloMOEOProbMatrixContainerUpdater.h>
#include <moeoNSGAII2.h>
#include <utils/moeoBestObjVecStat.h>
#include <utils/moeoAverageObjVecStat.h>
#include <PhyloMOEOPartitionStat.h>
#include <eoCountedFileMonitor.h>
#include <eoSingleFileCountedStateSaver.h>
#include <vectorSortIndex.h>
#include <utils.h>
#include <ctime>
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


int main(int argc, char *argv[])
{
	welcome_message();

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


	if( datafile.size()==0 )
	{
	 	parser.printHelp( cout );
		return(-1);
	}
	
	string filename = path + datafile;
	cout << "\n\nReading Sequence Datafile...";
	Sequences seq(filename.c_str());
	cout << " done.\n";
	// calculate datafile
	seq.calculate_patterns();
	seq.calculate_frequences();

 	rn2 = gsl_rng_alloc(gsl_rng_default);
 	rn = new RandomNr(seed);
	phylotreeIND templatetree( rn, seq, rn2);
	ParsimonyCalculator parsi_calc(templatetree);
 	SubstModel modelHKY( seq, SubstModel::HKY85);
 	modelHKY.init();
 	modelHKY.set_kappa(kappa); // banco_grande
 	ProbMatrixContainer probmatrixs(modelHKY);
	LikelihoodCalculator lik_calc(templatetree, modelHKY, probmatrixs,ncats);
	lik_calc.set_alpha(alpha);
 	modelHKY.init();
	PhyloEval byobj( parsi_calc, lik_calc );

	Phyloraninit initializer(templatetree);

	eoState state;
	
	//eoPop <PhyloMOEO> &population = state.takeOwnership(eoPop<PhyloMOEO>(popsize, initializer));

	eoPop<PhyloMOEO> population(popsize, initializer);
	//state.registerObject( population ); 

	cout << "\n\nReading Initial Trees...";
	if( usertree.size() >0)	
	{
		filename = path + usertree;
		readtrees(filename.c_str(), population);
	}
	cout << " done.\n";


	cout << "\n\nCreating output files...";

	try{
		filename = path + datafile + "_exp_param_" + expid + ".txt";
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
			// create the moea	
	save_exp_params(exp_data);
	seq.save_seq_data(exp_data); 

	moeoAverageObjVecStat <PhyloMOEO> bestfit;
	moeoBestObjVecStat <PhyloMOEO> avgfit;
	eoPopStat<PhyloMOEO> popstats;

	eoCountedFileMonitor media_scores( 2, path + datafile + "_media_scores_" + expid + ".txt", "\t", true,true );
	media_scores.add( bestfit);
	media_scores.add( avgfit) ;

	eoCountedFileMonitor evolution_scores( 2, path + datafile + "_evolution_data_" + expid + ".txt", "\n", true,true );
	evolution_scores.add( popstats);


	//cout << "\n\nRunning NSGA-II ..." << endl;

	eoGenContinue<PhyloMOEO> continuator(ngenerations);
	eoCheckPoint<PhyloMOEO> cp(continuator);
 	eoValueParam<unsigned> generationCounter(0, "Gen.");
	eoIncrementor<unsigned> increment(generationCounter.value()); 
	cp.add(increment); 
	eoStdoutMonitor monitor(false);
	monitor.add(generationCounter); 
	cp.add(monitor); 
	Phylomutate mutator;
	Phylocross crossover;
	eoSequentialOp<PhyloMOEO> operadores;
	operadores.add(crossover,pcrossover);
	operadores.add(mutator,pmutation);
	PhyloMOEOProbMatrixContainerUpdater probmatrixupdater(probmatrixs);
	cp.add( bestfit);
	cp.add( avgfit);
	cp.add( media_scores);
	cp.add( evolution_scores );
	cp.add( popstats);
	cp.add( probmatrixupdater );
	
//	apply<PhyloMOEO> ( byobj, population );
//	population.printOn(cout);
	if(algotype == "ibea")
	{
		moeoAdditiveEpsilonBinaryMetric < ObjectiveVector > metric;
		moeoIBEA < PhyloMOEO > algo (cp, byobj, operadores, metric);
		cout << "\n\nRunning IBEA ..." << endl;	
		algo(population);
	}	
	else
	{
		moeoNSGAII < PhyloMOEO > algo (cp, byobj, operadores);
		cout << "\n\nRunning NSGA-II ..." << endl;	
		algo(population);

	}
	
	cout << "\nCalculating Final Solutions...";
	cout << "  done\n";

	PhyloMOEOFinalSolutionsArchive finalsolutions;
	finalsolutions.operator()(population);

	//remove_final_solutions( population );
	// optimize remaining solutions
	cout << "\nOptimizing tree branch lenghts...\n";
	
	optimize_solutions( finalsolutions, lik_calc );
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
	gsl_rng_free(rn2);
//	delete probmatrixs;
	delete rn;
	cout << "\nPhyloMOEA execution finishes !\n";
	return 0;
} 


