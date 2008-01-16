/*
* <main.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, INRIA, 2007
*
* Alexandru-Adrian Tantar, Clive Canape
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/


#include <peo>

// Specific libraries (TSP)
#include <param.h>
#include <route.h>
#include <route_eval.h>
#include <route_init.h>
#include <two_opt.h>
#include <two_opt_init.h>
#include <two_opt_next.h>
#include <two_opt_incr_eval.h>


int main( int __argc, char** __argv )
{

  /* In this lesson you will learn to use a multi-start.
   *
   * Thanks to this method, you can use several local searches together !!!
   *
   */

// Parameter
  const unsigned int POP_SIZE = 10;
  srand( time(NULL) );

// Initializing the ParadisEO-PEO environment
  peo :: init( __argc, __argv );
// Processing the command line specified parameters
  loadParameters( __argc, __argv );

// Define a Hill Climbing (you can choose an other local search)
// ie Lessons of ParadisEO - MO
  Route route;
  RouteInit init;
  init(route);
  RouteEval eval;
  eval(route);
  TwoOptInit initHC;
  TwoOptNext nextHC;
  TwoOptIncrEval incrHC;
  moBestImprSelect< TwoOpt > selectHC;
  moHC< TwoOpt > hc(initHC, nextHC, incrHC, selectHC, eval);

// Define a population
  RouteInit initPop;	// Creates random Route objects
  RouteEval evalPop;	// Offers a fitness value for a specified Route object
  eoPop < Route > pop(POP_SIZE, initPop);
  for ( unsigned int index = 0; index < POP_SIZE; index++ )
    evalPop( pop[ index ] );

// Setting up the parallel wrapper
  peoSynchronousMultiStart< Route > parallelHC(hc);
  peoParallelAlgorithmWrapper WrapHC (parallelHC, pop);
  parallelHC.setOwner( WrapHC );

  peo :: run( );
  peo :: finalize( );
  if ( getNodeRank() == 1 )
    {

      std :: cout << "\n\nBefore : \n" << route;
      std :: cout << "\n\nWith the synchronous Multi-Start HCs:";
      for ( unsigned int index = 0; index < POP_SIZE; index++ )
        std::cout <<"\n"<< pop[ index ];
    }

}
