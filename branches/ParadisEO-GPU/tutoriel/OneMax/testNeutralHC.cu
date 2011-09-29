/*
  <testNeutralHC.cu>
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
// OneMax increment evaluation function
#include <problems/eval/OneMaxIncrEval.h>
// One Max solution
#include <GPUType/moGPUBitVector.h>
// Bit neighbor
#include <neighborhood/moGPUBitNeighbor.h>
// Ordered neighborhood
#include <neighborhood/moGPUOrderNeighborhoodByModif.h>
// The Solution and neighbor comparator
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
// The continuator
#include <continuator/moTrueContinuator.h>
// Neutral HC algorithm
#include <algo/moNeutralHC.h>
//To compute execution time
#include <performance/moGPUTimer.h>

//------------------------------------------------------------------------------------
// Define types of the representation solution, different neighbors and neighborhoods
//------------------------------------------------------------------------------------

typedef moGPUBitVector<eoMaximizingFitness> solution;
typedef moGPUBitNeighbor <eoMaximizingFitness> Neighbor;
typedef moGPUOrderNeighborhoodByModif<Neighbor> Neighborhood;

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

 //nbStep maximum step to do
  eoValueParam<unsigned int> nbStepParam(10, "nbStep", "maximum number of step", 'n');
  parser.processParam( nbStepParam, "numberStep" );
  unsigned nbStep = nbStepParam.value();
 
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

  Neighborhood neighborhood(SIZE,gpuEval);

  /* =========================================================
   *
   * The neutral Hill Climbing algorithm
   *
   * ========================================================= */
  //True continuator <=> Always continue

  moTrueContinuator<Neighbor> continuator;

  moNeutralHC<Neighbor> neutralHC(neighborhood,eval,gpuEval,nbStep,continuator);

  /* =========================================================
   *
   * Execute the neutral Hill climbing from random sollution
   *
   * ========================================================= */
  
  eval(sol);

  std::cout << "initial: " << sol<< std::endl;
  moGPUTimer timer;
  timer.start();
  neutralHC(sol);
  timer.stop();
  std::cout << "final:   " << sol << std::endl;
  printf("Execution time = %.2lf s\n",timer.getTime());

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
