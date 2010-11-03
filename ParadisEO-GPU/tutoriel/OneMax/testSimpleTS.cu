//Init the number of threads per block
#define BLOCK_SIZE 128

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
#include <neighborhood/moCudaOrderNeighborhood.h>
// The Solution and neighbor comparator
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
// The time continuator
#include <continuator/moTimeContinuator.h>
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



// REPRESENTATION

typedef moCudaBitVector<eoMaximizingFitness> solution;
typedef moCudaBitNeighbor <solution,eoMaximizingFitness> Neighbor;
typedef moCudaOrderNeighborhood<Neighbor> Neighborhood;


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
  eoValueParam<unsigned int> vecSizeParam(8, "vecSize", "Genotype size", 'V');
  parser.processParam( vecSizeParam, "Representation" );
  unsigned vecSize = vecSizeParam.value();

  // size tabu list
  eoValueParam<unsigned int> sizeTabuListParam(7, "sizeTabuList", "size of the tabu list", 'T');
  parser.processParam( sizeTabuListParam, "Search Parameters" );
  unsigned sizeTabuList = sizeTabuListParam.value();

  // time Limit
  eoValueParam<unsigned int> timeLimitParam(1, "timeLimit", "time limits", 't');
  parser.processParam( timeLimitParam, "Search Parameters" );
  unsigned timeLimit = timeLimitParam.value();

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

  moNeighborComparator<Neighbor> comparator;
  moSolNeighborComparator<Neighbor> solComparator;

  /* =========================================================
   *
   * a solution neighborhood
   *
   * ========================================================= */

  Neighborhood neighborhood(vecSize,cueval);
    
  /* =========================================================
   *
   * continuator
   *
   * ========================================================= */

  moTimeContinuator <Neighbor> continuator(timeLimit);

  /* =========================================================
   *
   * tabu list
   *
   * ========================================================= */

  //moNeighborVectorTabuList<shiftNeighbor> tl(sizeTabuList,0);
  // tabu list
  moNeighborVectorTabuList<Neighbor> tl(sizeTabuList,0);

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

  //Basic Constructor
  moTS<Neighbor> localSearch2(neighborhood,eval, cueval,  2, 7);

  //Simple Constructor
  moTS<Neighbor> localSearch3(neighborhood, eval, cueval, 5, tl);

  //General Constructor
  moTS<Neighbor> localSearch4(neighborhood, eval, cueval, comparator, solComparator, continuator, tl, inten, div, asp);

  /* =========================================================
   *
   * Execute the local search(TS) from random sollution
   *
   * ========================================================= */
  //Initilisation of the solution
  solution sol1(vecSize);
  eval(sol1);
  std::cout << "Tabu Search 1:" << std::endl;
  std::cout << "---------------------" << std::endl;
  std::cout << "initial: " << sol1.fitness()<< std::endl;
  moCudaTimer timer1;
  timer1.start();
  localSearch1(sol1);
  timer1.stop();
  printf("CUDA execution time = %f ms\n",timer1.getTime());
  timer1.deleteTimer();
  std::cout << "final:   " << sol1.fitness() << std::endl<<std::endl;
  /* =========================================================
   *
   * Execute the TS Basic Constructor 
   *
   * ========================================================= */
  solution sol2(vecSize);
  eval(sol2);
  std::cout << "Tabu Search 2:" << std::endl;
  std::cout << "---------------------" << std::endl;
  std::cout << "initial: " << sol2.fitness()<< std::endl;
  moCudaTimer timer2;
  timer2.start();
  localSearch2(sol2);
  timer2.stop();
  printf("CUDA execution time = %f ms\n",timer2.getTime());
  timer2.deleteTimer();
  std::cout << "final:   " << sol2.fitness() << std::endl<< std::endl;
  /* =========================================================
   *
   * Execute the TS Simple Constructor
   *
   * ========================================================= */
  solution sol3(vecSize);
  eval(sol3);
  std::cout << "Tabu Search 3:" << std::endl;
  std::cout << "---------------------" << std::endl;
  std::cout << "initial: " << sol3.fitness()<< std::endl;
  moCudaTimer timer3;
  timer3.start();
  localSearch3(sol3);
  timer3.stop();
  printf("CUDA execution time = %f ms\n",timer3.getTime());
  timer3.deleteTimer();
  std::cout << "final:   " << sol3.fitness() << std::endl<< std::endl;
  /* =========================================================
   *
   * Execute the TS General Constructor
   *
   * ========================================================= */
  solution sol4(vecSize);
  eval(sol4);
  std::cout << "Tabu Search 4:" << std::endl;
  std::cout << "---------------------" << std::endl;
  std::cout << "initial: " << sol4.fitness()<< std::endl;
  moCudaTimer timer4;
  timer4.start();
  localSearch4(sol4);
  timer4.stop();
  printf("CUDA execution time = %f ms\n",timer4.getTime());
  timer4.deleteTimer();
  std::cout << "final:   " << sol4.fitness() << std::endl<< std::endl;
    
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

