#include <eo>
#include <moeo>
#include <PhyloMOEO.h>
#include <PhyloMOEO_init.h>
#include <PhyloMOEO_eval.h>
#include <PhyloMOEO_archive.h>
#include <PhyloMOEOProbMatrixContainerUpdater.h>
#include <utils/moeoBestObjVecStat.h>
#include <utils/moeoAverageObjVecStat.h>
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
	int nexp = parser.createParam(10, "nexps", "NumExps", 'e',"Param").value();
	kappa = parser.createParam((double)(4.0), "kappa", "Kappa value", 'k',"Param").value();
	alpha = parser.createParam((double)(2.0), "alpha", "Alpha value", 'a',"Param").value();
	ncats = parser.createParam((unsigned int)(4), "nCat", "Number of Categories", 'r',"Param").value();
	datafile = parser.createParam(string(), "data", "Datafile", 'd',"Param").value();
	path = parser.createParam(string(), "path", "Treefile", 'p',"Param").value();

	cout << "\n\nReading Sequence Datafile..." << path+datafile;

	datafile = path+datafile;
	Sequences seq6(datafile.c_str());
//	Sequences seq7("/home/wcancino/experimentos/PhyloMOEA_0.2/omp_tests/datasets/TEST.500_5000");
	cout << " done.\n";
	// calculate datafile
	cout << "calculating pattersn..." << endl;

	seq6.calculate_patterns();
	seq6.calculate_frequences();


 	gsl_rng *rn2 = gsl_rng_alloc(gsl_rng_default);
 	RandomNr *rn = new RandomNr(time(NULL));
	omp_set_num_threads(nthreads);
	phylotreeIND templatetree6( rn, seq6, rn2);
	SubstModel modelHKY6( seq6, SubstModel::HKY85);
	modelHKY6.init();
	ProbMatrixContainer probmatrixs6(modelHKY6);
	LikelihoodCalculator lik_calc6(templatetree6, modelHKY6, probmatrixs6, 4);
	lik_calc6.set_alpha(alpha);
	modelHKY6.init();
 	modelHKY6.set_kappa(kappa); 
	Phyloraninit initializer6(templatetree6);

	PhyloMOEOParetoSolutionsArchive pareto_solutions;
	ParsimonyCalculator parsi_calc6(templatetree6);
	PhyloEval byobj( parsi_calc6, lik_calc6 );

	probmatrixs_ptr = &probmatrixs6;

	moeoBestObjVecStat <PhyloMOEO> bestfit;
	eoFileMonitor statfile(datafile + "_total_stats.txt");
	statfile.add(bestfit);
	for(int i=0; i < 2*nexp; i++)
	{
		eoPop<PhyloMOEO> population6(60, initializer6);

		ostringstream os;
		if(i<nexp) os << datafile << "_pareto_trees_serial_ibea";
		else os << datafile << "_pareto_trees_serial_nsga";
		os.fill('0');
		os.width(2);
		if(i<nexp) os << i+1 << ".txt";
		else os << i+1-nexp << ".txt";
		cout << "reading " << os.str() << endl;
		readtrees(os.str().c_str(),population6);
		//ofstream if(os.str().c_str());
		cout << "filtering solutions....." << endl;
		apply<PhyloMOEO> (byobj, population6);
		pareto_solutions.operator()(population6);

		bestfit.operator()(population6);
		statfile.operator()();
		cout.precision(15);
	}
	pareto_solutions.save_trees( datafile+"_total_pareto_trees.txt");
	pareto_solutions.save_scores( datafile+"_total_pareto_scores.txt");

//	of.close();
	gsl_rng_free(rn2);
	//	delete probmatrixs;
	delete rn;
	return 0;
}