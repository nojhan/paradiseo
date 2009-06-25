//
// C++ Implementation: testomp
//
// Description: 
//
//
// Author:  <>, (C) 2009
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include <eo>
#include <moeo>
#include <peo>
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
#include <omp.h>
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
	eoParser parser(argc, argv);
	datafile = parser.createParam(string(), "data", "Datafile", 'd',"Param").value();
	int nthreads = parser.createParam(omp_get_max_threads(), "nthreads", "Numthreads", 't',"Param").value();
	int ntrees = parser.createParam(20, "ntrees", "NumTrees", 'n',"Param").value();
	int nexp = parser.createParam(1, "nexps", "NumExps", 'e',"Param").value();

	cout << "\n\nReading Sequence Datafile...";

	Sequences seq6(datafile.c_str());
//	Sequences seq7("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/TEST.500_5000");
	cout << " done.\n";
	// calculate datafile
	cout << "calculating pattersn..." << endl;

	seq6.calculate_patterns();
	seq6.calculate_frequences();

	ostringstream os;
	os << datafile << "_results_serial_" << nthreads << ".txt";
 	ofstream of(os.str().c_str());

 	gsl_rng *rn2 = gsl_rng_alloc(gsl_rng_default);
 	RandomNr *rn = new RandomNr(time(NULL));
	omp_set_num_threads(nthreads);
	for(int i=0; i < nexp; i++)
	{
		phylotreeIND templatetree6( rn, seq6, rn2);
		SubstModel modelHKY6( seq6, SubstModel::HKY85);
		modelHKY6.init();
		ProbMatrixContainer probmatrixs6(modelHKY6);
		LikelihoodCalculator lik_calc6(templatetree6, modelHKY6, probmatrixs6, 4);
		modelHKY6.init();
		Phyloraninit initializer6(templatetree6);
		eoPop<PhyloMOEO> population6(ntrees, initializer6);
		cout.precision(15);
		PhyloLikelihoodTimeEval eval( lik_calc6 );


		cout << " Number of processors available:" << omp_get_num_procs() << " MAX Number of threads " << omp_get_max_threads() << endl;
		gettimeofday(&tempo1, NULL);
	
		apply<PhyloMOEO> (eval, population6);
	
		gettimeofday(&tempo2, NULL);
		cout << "\n"; print_elapsed_time(&tempo1,&tempo2);
		print_elapsed_time_short(&tempo1,&tempo2,of);
		of << endl;
	}
	of.close();
	gsl_rng_free(rn2);
	//	delete probmatrixs;
	delete rn;
	return 0;
}
	