/*
  <testSimpleTS_PPP.cu>
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

#include "moGPUConfig.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

__device__ int * dev_a;
__device__ int * dev_h;

// The general include for eo
#include <eo>
#include <ga.h>
// Fitness function
#include <problems/eval/moGPUPPPEval.h>
// GPU Fitness function
#include <eval/moGPUMappingEvalByModif.h>
#include <problems/eval/moGPUPPPIncrEval.h>
//Specific data to PPP problem
#include <problems/data/moGPUPPPData.h>
// PPP solution
#include <problems/types/moGPUPPPSolution.h>
// PPP neighbor
#include <problems/neighborhood/moGPUPPPNeighbor.h>
//To compute execution time
#include <performance/moGPUTimer.h>
//Utils to compute size Mapping of x-change position
#include <neighborhood/moGPUNeighborhoodSizeUtils.h>
//x-Change neighborhood
#include <neighborhood/moGPUXChangeNeighborhoodByModif.h>
// The Solution and neighbor comparator
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
// The Iter continuator
#include <continuator/moIterContinuator.h>
// The Tabou Search algorithm 
#include <algo/moTS.h>
//Tabu list
#include <memory/moIndexedVectorTabuList.h>
//Memories
#include <memory/moDummyIntensification.h>
#include <memory/moDummyDiversification.h>
#include <memory/moBestImprAspiration.h>

typedef moGPUPPPSolution<eoMinimizingFitness> solution;
typedef moGPUPPPNeighbor<solution> Neighbor;
typedef moGPUXChangeNeighborhoodByModif<Neighbor> Neighborhood;


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

    // Iteration number
    eoValueParam<unsigned int> nbIterationParam(100, "nbIteration", "TS Iteration number", 'I');
    parser.processParam( nbIterationParam, "TS Iteration number" );
    unsigned nbIteration = nbIterationParam.value();

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
   * Initilisation of PPP data
   *
   * ========================================================= */

  moGPUPPPData<int> _data;
  _data.load();

  /* =========================================================
   *
   * Initilisation of the solution
   *
   * ========================================================= */
 
   solution sol(Nd);
  _data.GPUObject.memCopyGlobalVariable(dev_a,_data.a_d);
  _data.GPUObject.memCopyGlobalVariable(dev_h,_data.H_d);
  
  /*=========================================================
   *
   * Evaluation of a solution neighbor's
   *
   * ========================================================= */

    moGPUPPPEval<solution> eval(_data);
    unsigned long int sizeMap=sizeMapping(Nd,NB_POS);
    moGPUPPPIncrEval<Neighbor> incr_eval;
    moGPUMappingEvalByModif<Neighbor,moGPUPPPIncrEval<Neighbor> > gpueval(sizeMap,incr_eval);
  
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

     Neighborhood neighborhood(sizeMap,NB_POS,gpueval);

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

     unsigned sizeTabuList=sizeMap;
     unsigned duration=sizeTabuList/8;
     // tabu list
     moIndexedVectorTabuList<Neighbor> tl(sizeTabuList,duration);

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
   * The Tabu search algorithm
   *
   * ========================================================= */

  moTS<Neighbor> tabuSearch(neighborhood, eval, gpueval, comparator, solComparator, continuator, tl, inten, div, asp);  
 
  /* =========================================================
   *
   * Execute the local search from random sollution
   *
   * ========================================================= */
 
  //Can be eval here, else it will be done at the beginning of the localSearch
  eval(sol);

  std::cout << "initial: " << sol<< std::endl;
  moGPUTimer timer;
  timer.start();
  tabuSearch(sol);
  std::cout << "final:   " << sol << std::endl;
  timer.stop();
  printf("Execution time = %.2lf s\n",timer.getTime());
 

  _data.GPUObject.free(dev_a);
  _data.GPUObject.free(dev_h);

  return 0;
}
