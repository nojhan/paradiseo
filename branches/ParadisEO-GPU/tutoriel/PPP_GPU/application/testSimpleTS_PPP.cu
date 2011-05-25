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
#include <problems/eval/PPPEval.h>
// GPU Fitness function
#include <eval/moGPUMappingEvalByModif.h>
#include <problems/eval/PPPIncrEval.h>
//Specific data to PPP problem
#include <problems/data/PPPData.h>
// PPP solution
#include <problems/types/PPPSolution.h>
// PPP neighbor
#include <problems/neighborhood/PPPNeighbor.h>
//To compute execution time
#include <performance/moGPUTimer.h>
//Utils to compute size Mapping of x-change position
#include <neighborhood/moNeighborhoodSizeUtils.h>
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

typedef PPPSolution<eoMinimizingFitness> solution;
typedef PPPNeighbor<solution> Neighbor;
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

   
    //Number of position to change 
    eoValueParam<unsigned int> nbPosParam(1, "nbPos", "X Change", 'N');
    parser.processParam( nbPosParam, "Exchange" );
    unsigned nbPos = nbPosParam.value();

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
//   rng.reseed(seed);
srand(0);
  /* =========================================================
   *
   * Initilisation of QAP data
   *
   * ========================================================= */

  PPPData<int> _data;
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

    PPPEval<solution> eval(_data);
    unsigned long int sizeMap=sizeMapping(Nd,NB_POS);
    PPPIncrEval<Neighbor> incr_eval;
    moGPUMappingEvalByModif<Neighbor,PPPIncrEval<Neighbor> > cueval(sizeMap,incr_eval);
  
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

     Neighborhood neighborhood(sizeMap,NB_POS,cueval);

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

     sizeTabuList=sizeMap;
     duration=sizeTabuList/8;
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

  moTS<Neighbor> tabuSearch(neighborhood, eval, cueval, comparator, solComparator, continuator, tl, inten, div, asp);  
 
  /* =========================================================
   *
   * Execute the local search from random sollution
   *
   * ========================================================= */
 
  //Can be eval here, else it will be done at the beginning of the localSearch
  eval(sol);

  std::cout << "initial: " << sol<< std::endl;
  // Create timer for timing CUDA calculation
  /*cudaFuncSetCacheConfig(moGPUMappingKernelEvalByModif<int,eoMinimizingFitness,PPPIncrEval<Neighbor> >,  cudaFuncCachePreferL1);*/
  moGPUTimer timer;
  timer.start();
  tabuSearch(sol);
  std::cout << "final:   " << sol << std::endl;
  timer.stop();
  printf("CUDA execution time = %f ms\n",timer.getTime());
  timer.deleteTimer();
 

  _data.GPUObject.free(dev_a);
  _data.GPUObject.free(dev_h);

  return 0;
}
