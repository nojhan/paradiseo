///Init the number of threads per block
#define BLOCK_SIZE 128

//-----------------------------------------------------------------------------
#include <iostream>  
#include <stdlib.h> 
using namespace std;

// The general include for eo
#include <eo>
#include <ga.h>
// Fitness function
#include <problems/eval/EvalOneMax.h>
// Cuda Fitness function
#include <eval/moCudaVectorEval.h>
#include <problems/eval/OneMaxIncrEval.h>
// One Max solution
#include <cudaType/moCudaBitVector.h>
//To compute execution time
#include <performance/moCudaTimer.h>
// One Max neighbor
#include <neighborhood/moCudaBitNeighbor.h>
// One Max ordered neighborhood
#include <neighborhood/moCudaRndWithReplNeighborhood.h>
//Algorithm and its components
#include <coolingSchedule/moCoolingSchedule.h>
#include <algo/moSA.h>

// The simulated annealing algorithm explorer
#include <explorer/moSAexplorer.h>

//comparator
#include <comparator/moSolNeighborComparator.h>

//continuators
#include <continuator/moTrueContinuator.h>
#include <continuator/moCheckpoint.h>
#include <continuator/moFitnessStat.h>
#include <utils/eoFileMonitor.h>
#include <continuator/moCounterMonitorSaver.h>

//-----------------------------------------------------------------------------
// Define types of the representation solution, different neighbors and neighborhoods
//-----------------------------------------------------------------------------
// REPRESENTATION

typedef moCudaBitVector<eoMaximizingFitness> solution;
typedef moCudaBitNeighbor <solution,eoMaximizingFitness> Neighbor;
typedef moCudaRndWithReplNeighborhood<Neighbor> Neighborhood;

void main_function(int argc, char **argv)
{
	/* =========================================================
*
* Parameters
*
* ========================================================= */

// First define a parser from the command-line arguments
eoParser parser(argc, argv);

// For each parameter, define Parameter, read it through the parser,
// and assign the value to the variable

eoValueParam<uint32_t> seedParam(time(0), "seed", "Random number seed", 'S');
parser.processParam( seedParam );
unsigned seed = seedParam.value();

// description of genotype
eoValueParam<unsigned int> vecSizeParam(8, "vecSize", "Genotype size", 'V');
parser.processParam( vecSizeParam, "Representation" );
unsigned vecSize = vecSizeParam.value();

// the name of the "status" file where all actual parameter values will be saved
string str_status = parser.ProgramName() + ".status"; // default value
eoValueParam<string> statusParam(str_status.c_str(), "status", "Status file");
parser.processParam( statusParam, "Persistence" );

// do the following AFTER ALL PARAMETERS HAVE BEEN PROCESSED
// i.e. in case you need parameters somewhere else, postpone these
if (parser.userNeedsHelp()) {
	parser.printHelp(cout);
	exit(1);
}
if (statusParam.value() != "") {
	ofstream os(statusParam.value().c_str());
	os << parser;// and you can use that file as parameter file
}

/* =========================================================
 *
 * Random seed
 *
 * ========================================================= */

//reproducible random seed: if you don't change SEED above,
// you'll always get the same result, NOT a random run
rng.reseed(seed);

/* =========================================================
 *
 * Initilisation of the solution
 *
 * ========================================================= */

//solution sol(size);
solution sol(vecSize);

for(unsigned int i=0;i<vecSize;i++) cout<<sol[i]<<"  ";
cout<<endl;

/* =========================================================
 *
 * Eval fitness function
 *
 * ========================================================= */

EvalOneMax<solution> eval;

/* =========================================================
 *
 * Evaluation of a solution neighbor's
 *
 * ========================================================= */

  OneMaxIncrEval<Neighbor> incr_eval;
  moCudaVectorEval<Neighbor,OneMaxIncrEval<Neighbor> > cueval(vecSize,incr_eval);
/* =========================================================
 *
 * Comparator of solutions and neighbors
 *
 * ========================================================= */

 // moNeighborComparator<Neighbor> comparator;
  moSolNeighborComparator<Neighbor> solComparator;

/* =========================================================
 *
 * a solution neighborhood
 *
 * ========================================================= */

  Neighborhood neighborhood(vecSize,cueval);
/* =========================================================
 *
 * the cooling schedule of the process
 *
 * ========================================================= */


// initial temp, factor of decrease, number of steps without decrease, final temp.
  moSimpleCoolingSchedule<solution> coolingSchedule(500, 0.9, 1000, 0.01);

/* =========================================================
 *
 * the local search algorithm
 *
 * ========================================================= */

  moSA<Neighbor> localSearch1(neighborhood, eval, cueval,coolingSchedule);

/* =========================================================
 *
 * execute the local search from random solution
 *
 * ========================================================= */

//init(solution);
eval(sol);

std::cout << "#########################################" << std::endl;
std::cout << "initial solution1: " << sol.fitness() << std::endl;

localSearch1(sol);

std::cout << "final solution1: " << sol.fitness() << std::endl;
std::cout << "#########################################" << std::endl;


/* =========================================================
 *
 * Comparator of neighbors
 *
 * ========================================================= */

// moSolNeighborComparator<Neighbor> solComparator;

/* =========================================================
 *
 * Example of Checkpointing
 *
 * ========================================================= */

 /*moTrueContinuator<Neighbor> continuator;//always continue
 moCheckpoint<Neighbor> checkpoint(continuator);
 moFitnessStat<solution> fitStat;
 checkpoint.add(fitStat);
 eoFileMonitor monitor("fitness.out", "");
 moCounterMonitorSaver countMon(100, monitor);
 checkpoint.add(countMon);
 monitor.add(fitStat);

 moSA<Neighbor> localSearch2(neighborhood, eval, cueval, coolingSchedule, solComparator, checkpoint);

 
 eval(sol1);
 for(int i=0;i<vecSize;i++) cout<<sol1[i]<<"  ";
 cout<<endl;

 std::cout << "#########################################" << std::endl;
 std::cout << "initial solution2: " << sol1.fitness() << std::endl ;

 localSearch2(sol);

 std::cout << "final solution2: " << sol1.fitness() << std::endl ;
 std::cout << "#########################################" << std::endl;*/
}

// A main that catc hes the exceptions

int main(int argc, char **argv)
{
	try {
		main_function(argc, argv);
	}
	catch (exception& e) {
		cout << "Exception: " << e.what() << '\n';
	}
	return 1;
}
