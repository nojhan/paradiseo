//Init the number of threads per block
#define BLOCK_SIZE 512
#include <iostream>  
#include <stdlib.h> 
using namespace std;

// The general include for eo
#include <eo>
#include <ga.h>
// Fitness function
#include <problems/eval/EvalOneMax.h>
// Cuda Fitness function
#include <eval/moCudaKswapEval.h>
#include <problems/eval/OneMaxIncrEval.h>
// One Max solution
#include <cudaType/moCudaBitVector.h>
// One Max neighbor
#include <neighborhood/moBitFlippingNeighbor.h>
//To compute execution time
#include <performance/moCudaTimer.h>
// One Max ordered neighborhood
#include <neighborhood/moCudaKswapNeighborhood.h>
// The Solution and neighbor comparator
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
// The continuator
#include <continuator/moTrueContinuator.h>
// Local search algorithm
#include <algo/moLocalSearch.h>
// Simple HC algorithm
#include <algo/moSimpleHC.h>
// The simple HC algorithm explorer
#include <explorer/moSimpleHCexplorer.h>

/**
 * @return the factorial of an unsigned integer
 * @param i an integer
 */

unsigned long int factorial1(unsigned int i) {
	if (i == 0)
		return 1;
	else
		return i * factorial1(i - 1);
}

/**
 * @return the neighborhood Size from the solution size and number of swap
 * @param _size the solution size
 * @param _Kswap the number of swap
 */

unsigned long int sizeMapping1( unsigned int _size, unsigned int _Kswap) {

	unsigned long int _sizeMapping = _size;
	for (unsigned int i = _Kswap; i > 0; i--) {
		_sizeMapping *= (_size - i);
	}
	_sizeMapping /= factorial1(_Kswap + 1);
	return _sizeMapping;
}

// REPRESENTATION
typedef moCudaBitVector<eoMaximizingFitness> solution;
typedef moBitFlippingNeighbor<solution> Neighbor;
typedef moCudaKswapNeighborhood<Neighbor> Neighborhood;

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

  // seed
  eoValueParam<uint32_t> seedParam(time(0), "seed", "Random number seed", 'S');
  parser.processParam( seedParam );
  unsigned seed = seedParam.value();

  // description of genotype
  eoValueParam<unsigned int> vecSizeParam(6, "vecSize", "Genotype size", 'V');
  parser.processParam( vecSizeParam, "Representation" );
  unsigned vecSize = vecSizeParam.value();

// Swap number
  eoValueParam<unsigned int> KSwapParam(0, "KSwap", "swap number", 'N');
  parser.processParam(KSwapParam, "KSwap" );
  unsigned KSwap = KSwapParam.value();

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
  // you'll aways get the same result, NOT a random run
  rng.reseed(seed);

  
  /* =========================================================
   *
   * Initilisation of the solution
   *
   * ========================================================= */
  
  solution sol(vecSize);
  if(vecSize<64)
    for(unsigned i=0;i<vecSize;i++) cout<<sol[i]<<"  ";
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
  unsigned long int sizeMap=sizeMapping1(vecSize,KSwap);
  std::cout<<"sizeMap : "<<sizeMap<<std::endl;
  OneMaxIncrEval<Neighbor> incr_eval;
  moCudaKswapEval<Neighbor,OneMaxIncrEval<Neighbor> > cueval(sizeMap,incr_eval);
  
  /* =========================================================
   *
   * Comparator of solutions and neighbors
   *
   * ========================================================= */

  moNeighborComparator<Neighbor> comparator;
  moSolNeighborComparator<Neighbor> solComparator;

  /* =========================================================
   *
   * a solution neighborhood
   *
   * ========================================================= */

  Neighborhood neighborhood(vecSize,KSwap,cueval);

  /* =========================================================
   *
   * An explorer of solution neighborhood's
   *
   * ========================================================= */

  moSimpleHCexplorer<Neighbor> explorer(neighborhood, cueval,
					comparator, solComparator);

  /* =========================================================
   *
   * The local search algorithm
   *
   * ========================================================= */
  //True continuator <=> Always continue

  moTrueContinuator<Neighbor> continuator;

  moLocalSearch<Neighbor> localSearch(explorer,continuator, eval);

  /* =========================================================
   *
   * The simple Hill Climbing algorithm
   *
   * ========================================================= */

  moSimpleHC<Neighbor> simpleHC(neighborhood,eval,cueval);

  /* =========================================================
   *
   * Execute the local search from random sollution
   *
   * ========================================================= */
 
  //Can be eval here, else it will be done at the beginning of the localSearch
  eval(sol);

  std::cout << "initial: " << sol.fitness()<< std::endl;
  // Create timer for timing CUDA calculation
  moCudaTimer timer;
  timer.start();
  localSearch(sol);
  timer.stop();
  printf("CUDA execution time = %f ms\n",timer.getTime());
  timer.deleteTimer();
  std::cout << "final:   " << sol.fitness() << std::endl;

  
}

// A main that catches the exceptions

int main(int argc, char **argv)
{
  try{
    main_function(argc, argv);
  }
  catch(exception& e){
    cout << "Exception: " << e.what() << '\n';
  }
  return 1;
}
