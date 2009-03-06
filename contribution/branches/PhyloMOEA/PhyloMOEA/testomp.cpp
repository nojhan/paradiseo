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


	cout << "\n\nReading Sequence Datafile...";
// 	Sequences seq("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/TEST.50_5000");
// 	Sequences seq2("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/TEST.50_50000");
// 	Sequences seq3("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/TEST.50_500000");

/*	Sequences seq4("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/TEST.250_5000");
	Sequences seq5("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/TEST.250_50000");*/
	Sequences seq6("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/TEST.250_500000");
//	Sequences seq7("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/TEST.500_5000");
	cout << " done.\n";
	// calculate datafile
	cout << "calculating pattersn..." << endl;

/*	seq.calculate_patterns();
	seq.calculate_frequences();
	seq2.calculate_patterns();
	seq2.calculate_frequences();
	seq3.calculate_patterns();
	seq3.calculate_frequences();*/
/*	seq4.calculate_patterns();
	seq4.calculate_frequences();
	seq5.calculate_patterns();
	seq5.calculate_frequences();*/
	seq6.calculate_patterns();
	seq6.calculate_frequences();
// 	seq7.calculate_patterns();
// 	seq7.calculate_frequences();


 	gsl_rng *rn2 = gsl_rng_alloc(gsl_rng_default);
 	RandomNr *rn = new RandomNr(time(NULL));

/*	phylotreeIND templatetree( rn, seq, rn2);
	phylotreeIND templatetree2( rn, seq2, rn2);
	phylotreeIND templatetree3( rn, seq3, rn2);*/
/*	phylotreeIND templatetree4( rn, seq4, rn2);
	phylotreeIND templatetree5( rn, seq5, rn2);*/
	phylotreeIND templatetree6( rn, seq6, rn2);
// 	phylotreeIND templatetree7( rn, seq7, rn2);


/* 	SubstModel modelHKY( seq, SubstModel::HKY85);
 	SubstModel modelHKY2( seq2, SubstModel::HKY85);
 	SubstModel modelHKY3( seq3, SubstModel::HKY85);*/
/* 	SubstModel modelHKY4( seq4, SubstModel::HKY85);
 	SubstModel modelHKY5( seq5, SubstModel::HKY85);*/
 	SubstModel modelHKY6( seq6, SubstModel::HKY85);
//  	SubstModel modelHKY7( seq7, SubstModel::HKY85);

/* 	modelHKY.init();
 	modelHKY2.init();
 	modelHKY3.init();*/
/* 	modelHKY4.init();
 	modelHKY5.init();*/
 	modelHKY6.init();
// 	modelHKY7.init();
 	//modelHKY.set_kappa(3.890); 
/* 	ProbMatrixContainer probmatrixs(modelHKY);
 	ProbMatrixContainer probmatrixs2(modelHKY2);
 	ProbMatrixContainer probmatrixs3(modelHKY3);*/
/* 	ProbMatrixContainer probmatrixs4(modelHKY4);
 	ProbMatrixContainer probmatrixs5(modelHKY5);*/
 	ProbMatrixContainer probmatrixs6(modelHKY6);
//  	ProbMatrixContainer probmatrixs7(modelHKY7);


	//probmatrixs_ptr = &probmatrixs;


/*	LikelihoodCalculator lik_calc(templatetree, modelHKY, probmatrixs, 4);
	LikelihoodCalculator lik_calc2(templatetree2, modelHKY2, probmatrixs2, 4);
	LikelihoodCalculator lik_calc3(templatetree3, modelHKY3, probmatrixs3, 4);*/
/*	LikelihoodCalculator lik_calc4(templatetree4, modelHKY4, probmatrixs4, 4);
	LikelihoodCalculator lik_calc5(templatetree5, modelHKY5, probmatrixs5, 4);	*/
	LikelihoodCalculator lik_calc6(templatetree6, modelHKY6, probmatrixs6, 4);
//	LikelihoodCalculator lik_calc7(templatetree7, modelHKY7, probmatrixs7, 4);
	//lik_calc.set_alpha(0.950);
/* 	modelHKY.init();
	modelHKY2.init();
	modelHKY3.init();*/
/*	modelHKY4.init();
	modelHKY5.init();*/
	modelHKY6.init();
// 	modelHKY7.init();



/*	Phyloraninit initializer(templatetree);
	Phyloraninit initializer2(templatetree2);
	Phyloraninit initializer3(templatetree3);*/
/*	Phyloraninit initializer4(templatetree4);
	Phyloraninit initializer5(templatetree5);*/
	Phyloraninit initializer6(templatetree6);
// 	Phyloraninit initializer7(templatetree7);

/*	eoPop<PhyloMOEO> population(1, initializer);
	eoPop<PhyloMOEO> population2(1, initializer2);
	eoPop<PhyloMOEO> population3(1, initializer3);*/
/*	eoPop<PhyloMOEO> population4(1, initializer4);
	eoPop<PhyloMOEO> population5(1, initializer5);*/
	eoPop<PhyloMOEO> population6(1, initializer6);
// 	eoPop<PhyloMOEO> population7(1, initializer7);

	cout << "Reading trees..." << endl;	
/*	readtrees("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/start.TEST.50_5000", population);
	readtrees("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/start.TEST.50_50000", population2);
	readtrees("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/start.TEST.50_500000", population3);*/
/*	readtrees("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/start.TEST.250_5000", population4);
	readtrees("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/start.TEST.250_50000", population5);*/
	readtrees("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/start.TEST.250_500000", population6);
// 	readtrees("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/start.TEST.500_5000", population7);

	gettimeofday(&tempo1, NULL);

	cout << " Number of processors available:" << omp_get_num_procs() << " MAX Number of threads " << omp_get_max_threads() << endl;

	cout.precision(15);
// 	lik_calc.set_tree(population[0].get_tree());
// 	cout << lik_calc.calculate_likelihood() << endl;
// 	cout << lik_calc.calculate_likelihood_omp() << endl;
// 
// 	lik_calc2.set_tree(population2[0].get_tree());
// 	cout << lik_calc2.calculate_likelihood() << endl;
// 	cout << lik_calc2.calculate_likelihood_omp() << endl;
// 
// 	lik_calc3.set_tree(population3[0].get_tree());
// 	cout << lik_calc3.calculate_likelihood() << endl;
// 	cout << lik_calc3.calculate_likelihood_omp() << endl;

// 	lik_calc4.set_tree(population4[0].get_tree());
// 	cout << lik_calc4.calculate_likelihood() << endl;
// 	cout << lik_calc4.calculate_likelihood_omp() << endl;
// 
// 	lik_calc5.set_tree(population5[0].get_tree());
// 	cout << lik_calc5.calculate_likelihood() << endl;
// 	cout << lik_calc5.calculate_likelihood_omp() << endl;

	lik_calc6.set_tree(population6[0].get_tree());
	cout << lik_calc6.calculate_likelihood() << endl;
	cout << lik_calc6.calculate_likelihood_omp() << endl;
// 
// 	lik_calc7.set_tree(population7[0].get_tree());
// 	cout << lik_calc7.calculate_likelihood() << endl;
// 	cout << lik_calc7.calculate_likelihood_omp() << endl;

	
/*
	for(int i=0; i < population.size(); i++)
	{
	
			lik_calc.set_tree(population[i].get_tree());
//			gettimeofday(&tempo1, NULL);
			cout << lik_calc.calculate_likelihood_omp() << endl;
/*			gettimeofday(&tempo2, NULL);
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
			cout << seconds << "." << result.tv_usec << "(" << result.tv_sec << ")" << endl;*/
//	}


	gsl_rng_free(rn2);
	//	delete probmatrixs;
	delete rn;
	return 0;
}
	