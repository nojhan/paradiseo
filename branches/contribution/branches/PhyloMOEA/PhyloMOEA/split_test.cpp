#include <eo>
#include <phylotreeIND.h>
using namespace std;
gsl_rng *rn2;
RandomNr *rn;
//Sequences *seq;
long seed;
//vector<phylotreeIND> arbores;
string datafile, path;
phylotreeIND *templatetree_ptr;


int main(int argc, char *argv[])
{
	// measures execution time
	eoParser parser(argc, argv);
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
	phylotreeIND templatetree6( rn, seq6, rn2);
	phylotreeIND *test = templatetree6.randomClone();
	phylotreeIND test2(*test);
	test->calculate_splits4();
	test2.calculate_splits4();
	test->export_subtree(test2);
	//test2.TBR();

	//test2.printNewick(cout);
	cout << "calculando splits..." << endl;
	test2.calculate_splits4();
	//test->print_splits_2();
	
	cout << "calculando distance..." << endl;
	cout << "distance " << test->compare_topology_4(test2) << endl;
	cout << "calculando distance..." << endl;
	cout << "distance " << test->compare_topology_2(test2) << endl;

	
//	of.close();
	gsl_rng_free(rn2);
	//	delete probmatrixs;
	delete rn;
	return 0;
}