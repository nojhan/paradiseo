/*
  <testSimulatedAnnealing.cu>
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
#include <iostream>  
#include <stdlib.h> 
using namespace std;

//Include GPU Config File
#include "moGPUConfig.h"
// The general include for eo
#include <eo>
#include <ga.h>
// OneMax full eval function
#include <problems/eval/EvalOneMax.h>
//Parallel evaluation of neighborhood on GPU
#include <eval/moGPUEvalByModif.h>
// OneMax increment eval function
#include <problems/eval/OneMaxIncrEval.h>
// One Max solution
#include <GPUType/moGPUBitVector.h>
// Bit neighbor
#include <neighborhood/moGPUBitNeighbor.h>
// Random with replacement neighborhood
#include <neighborhood/moGPURndWithReplNeighborhoodByModif.h>
// The Solution and neighbor comparator
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
//To compute execution time
#include <performance/moGPUTimer.h>
//Algorithm and its components
#include <coolingSchedule/moCoolingSchedule.h>
#include <algo/moSA.h>
// The simulated annealing algorithm explorer
#include <explorer/moSAexplorer.h>
//continuators
#include <continuator/moTrueContinuator.h>
#include <continuator/moCheckpoint.h>
#include <continuator/moFitnessStat.h>
#include <utils/eoFileMonitor.h>
#include <continuator/moCounterMonitorSaver.h>

//------------------------------------------------------------------------------------
// Define types of the representation solution, different neighbors and neighborhoods
//------------------------------------------------------------------------------------

typedef moGPUBitVector<eoMaximizingFitness> solution;
typedef moGPUBitNeighbor <eoMaximizingFitness> Neighbor;
typedef moGPURndWithReplNeighborhoodByModif<Neighbor> Neighborhood;

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

  solution sol(SIZE);

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
  moGPUEvalByModif<Neighbor,OneMaxIncrEval<Neighbor> > gpuEval(SIZE,incr_eval);

  /* =========================================================
   *
   * a solution neighborhood
   *
   * ========================================================= */

  Neighborhood neighborhood(SIZE,gpuEval);

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

  moSA<Neighbor> SA(neighborhood, eval, gpuEval,coolingSchedule);

  /* =========================================================
   *
   * execute the local search from random solution
   *
   * ========================================================= */

  //init(solution);
  eval(sol);
  std::cout << "initial : " << sol << std::endl;
  moGPUTimer timer;
  timer.start();
  SA(sol);
  timer.stop(); 
  std::cout << "final : " << sol << std::endl;
  printf("Execution time = %.2lf s\n",timer.getTime()); 

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
