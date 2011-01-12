//Init the number of threads per block
#define BLOCK_SIZE 256
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

__device__ int * dev_a;
__device__ int * dev_b;

// The general include for eo
#include <eo>
#include <ga.h>
// Fitness function
#include <problems/eval/QAPEval.h>
// Cuda Fitness function
#include <eval/moCudaKswapEval.h>
#include <problems/eval/QAPIncrEval.h>
//Specific data to QAP problem
#include <problems/data/QAPData.h>
// QAP solution
#include <cudaType/moCudaIntVector.h>
// QAP neighbor
#include <neighborhood/moKswapNeighbor.h>
//To compute execution time
#include <performance/moCudaTimer.h>
// QAP ordered neighborhood
#include <neighborhood/moCudaKswapNeighborhood.h>
// The Solution and neighbor comparator
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
// The Iter continuator
#include <continuator/moIterContinuator.h>
// Local search algorithm
#include <algo/moLocalSearch.h>
// The Tabou Search algorithm explorer
#include <explorer/moTSexplorer.h>
//Algorithm and its components
#include <algo/moTS.h>
//Tabu list
#include <memory/moNeighborVectorTabuList.h>
//Memories
#include <memory/moDummyIntensification.h>
#include <memory/moDummyDiversification.h>
#include <memory/moBestImprAspiration.h>
//#include <time.h>


typedef moCudaIntVector<eoMinimizingFitness> solution;
typedef moKswapNeighbor<solution> Neighbor;
typedef moCudaKswapNeighborhood<Neighbor> Neighborhood;


int main(int argc, char **argv)
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

 // Swap number
  eoValueParam<unsigned int> KSwapParam(1, "KSwap", "swap number", 'N');
  parser.processParam(KSwapParam, "KSwap" );
  unsigned KSwap = KSwapParam.value();

  // Iteration number
  eoValueParam<unsigned int> nbIterationParam(1, "nbIteration", "TS Iteration number", 'I');
  parser.processParam( nbIterationParam, "TS Iteration number" );
  unsigned nbIteration = nbIterationParam.value();

  // size tabu list
  eoValueParam<unsigned int> sizeTabuListParam(7, "sizeTabuList", "size of the tabu list", 'T');
  parser.processParam( sizeTabuListParam, "Search Parameters" );
  unsigned sizeTabuList = sizeTabuListParam.value();

  // duration tabu list
  eoValueParam<unsigned int> durationParam(7, "duration", "duration of the tabu list", 'D');
  parser.processParam( durationParam, "Search Parameters" );
  unsigned duration = durationParam.value();

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
  //srand(time(NULL));
  
  /* =========================================================
   *
   * Initilisation of QAP data
   *
   * ========================================================= */

   QAPData<int> _data(argv[1]);
   unsigned vecSize=_data.sizeData;
  /* =========================================================
   *
   * Initilisation of the solution
   *
   * ========================================================= */
 
   solution sol(vecSize);
  _data.cudaObject.memCopyGlobalVariable(dev_a,_data.a_d);
  _data.cudaObject.memCopyGlobalVariable(dev_b,_data.b_d);
  
 /* =========================================================
   *
   * Evaluation of a solution neighbor's
   *
   * ========================================================= */
  QAPEval<solution> eval(_data);
  unsigned long int sizeMap=sizeMapping(vecSize,KSwap);
//  std::cout<<"sizeMap : "<<sizeMap<<std::endl;
  QAPIncrEval<Neighbor> incr_eval;
  moCudaKswapEval<Neighbor,QAPIncrEval<Neighbor> > cueval(sizeMap,incr_eval);
  
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
   * continuator
   *
   * ========================================================= */

   moIterContinuator <Neighbor> continuator(nbIteration);

 /* =========================================================
   *
   * tabu list
   *
   * ========================================================= */

  //moNeighborVectorTabuList<shiftNeighbor> tl(sizeTabuList,0);
  sizeTabuList=(vecSize*(vecSize-1))/2;
  duration=sizeTabuList/8;
  // tabu list
  moNeighborVectorTabuList<Neighbor> tl(sizeTabuList,duration);

  /* =========================================================
   *
   * Memories
   *
   * ========================================================= */

  moDummyIntensification<Neighbor> inten;
  moDummyDiversification<Neighbor> div;
  moBestImprAspiration<Neighbor> asp;

  /* =========================================================
   *
   * An explorer of solution neighborhood's
   *
   * ========================================================= */

  moTSexplorer<Neighbor> explorer(neighborhood, cueval, comparator, solComparator, tl, inten, div, asp);

  
  /* =========================================================
   *
   * the local search algorithm
   *
   * ========================================================= */

  moLocalSearch<Neighbor> localSearch1(explorer, continuator, eval);

  /* =========================================================
   *
   * Execute the local search from random sollution
   *
   * ========================================================= */
 
  //Can be eval here, else it will be done at the beginning of the localSearch
  eval(sol);

  std::cout << "initial: " << sol<< std::endl;
  // Create timer for timing CUDA calculation
  moCudaTimer timer;
  timer.start();
  localSearch1(sol);
  std::cout << "final:   " << sol << std::endl;
  timer.stop();
  printf("CUDA execution time = %f ms\n",timer.getTime());
  timer.deleteTimer();
 

 _data.cudaObject.free(dev_a);
 _data.cudaObject.free(dev_b);

return 0;
}
