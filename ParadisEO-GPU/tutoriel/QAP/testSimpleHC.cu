/*
  <testSimpleHC.cu>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Karima Boufaras, Thé Van LUONG

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.

  ParadisEO WebSite : http://paradiseo.gforge.inria.fr
  Contact: paradiseo-help@lists.gforge.inria.fr
*/

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
// One Max neighbor
#include <neighborhood/moKswapNeighbor.h>
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
#include <time.h>


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
  if (argc < 2){
    printf("Saisissez le nom de fichier dat à manipuler \n");
    exit(1);
  }
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
  eoValueParam<unsigned int> KSwapParam(1, "KSwap", "swap number", 'N');
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
  srand(time(NULL));
  
  /* =========================================================
   *
   * Initilisation of QAP data
   *
   * ========================================================= */

  QAPData<int> _data(argv[1]);
  vecSize=_data.sizeData;
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

  std::cout << "initial: " << sol<< std::endl;
  // Create timer for timing CUDA calculation
  moCudaTimer timer;
  timer.start();
  localSearch(sol);
  std::cout << "final:   " << sol << std::endl;
  timer.stop();
  printf("CUDA execution time = %f ms\n",timer.getTime());
  timer.deleteTimer();

  _data.cudaObject.free(dev_a);
  _data.cudaObject.free(dev_b);

  return 0;
}
