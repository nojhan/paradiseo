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

	cout << "\n\nReading Sequence Datafile...";
	Sequences seq("/home/wcancino/experimentos/PhyloMOEA_0.2/500/500_ZILLA.plain");
	cout << " done.\n";
	// calculate datafile
	cout << "calculating pattersn..." << endl;
	seq.calculate_patterns();
	seq.calculate_frequences();

 	gsl_rng *rn2 = gsl_rng_alloc(gsl_rng_default);
 	RandomNr *rn = new RandomNr(time(NULL));
	phylotreeIND templatetree( rn, seq, rn2);
 	SubstModel modelHKY( seq, SubstModel::HKY85);
 	modelHKY.init();
 	modelHKY.set_kappa(3.890); 
 	ProbMatrixContainer probmatrixs(modelHKY);
	probmatrixs_ptr = &probmatrixs;
	LikelihoodCalculator lik_calc(templatetree, modelHKY, probmatrixs, 4);
	lik_calc.set_alpha(0.950);
 	modelHKY.init();

	Phyloraninit initializer(templatetree);
	eoPop<PhyloMOEO> population(100, initializer);
	cout << "Reading trees..." << endl;	
	readtrees("/home/wcancino/experimentos/PhyloMOEA_0.2/500/init_trees.dat", population);

	gettimeofday(&tempo1, NULL);

	//cout << " Number of processors available:" << omp_get_num_procs() << " MAX Number of threads " << omp_get_max_threads() << endl;

	for(int i=0; i < population.size(); i++)
	{
			lik_calc.set_tree(population[i].get_tree());
			cout << lik_calc.calculate_likelihood() << endl;
	}

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
	gsl_rng_free(rn2);
	//	delete probmatrixs;
	delete rn;
	return 0;
}
	