/*
* <main.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, INRIA, 2007
*
* Clive Canape
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

#include "param.h"
#include "route_init.h"
#include "route_eval.h"
#include "order_xover.h"
#include "edge_xover.h"
#include "partial_mapped_xover.h"
#include "city_swap.h"
#include "part_route_eval.h"
#include "merge_route_eval.h"
#include "two_opt_init.h"
#include "two_opt_next.h"
#include "two_opt_incr_eval.h"

#include <peo>

#define POP_SIZE 10
#define NUM_GEN 100
#define CROSS_RATE 1.0
#define MUT_RATE 0.01


int main (int __argc, char * * __argv)
{

  peo :: init (__argc, __argv);
  loadParameters (__argc, __argv); 
  RouteInit route_init;
  RouteEval full_eval;
  OrderXover order_cross; 
  PartialMappedXover pm_cross;
  EdgeXover edge_cross;
  CitySwap city_swap_mut;  

// Initialization of the local search
  TwoOptInit pmx_two_opt_init;
  TwoOptNext pmx_two_opt_next;
  TwoOptIncrEval pmx_two_opt_incr_eval;
  moBestImprSelect <TwoOpt> pmx_two_opt_move_select;
  moHC <TwoOpt> hc (pmx_two_opt_init, pmx_two_opt_next, pmx_two_opt_incr_eval, pmx_two_opt_move_select, full_eval);

// EA
  eoPop <Route> pop (POP_SIZE, route_init);
  eoGenContinue <Route> cont (NUM_GEN); 
  eoCheckPoint <Route> checkpoint (cont);
  eoEvalFuncCounter< Route > eval(full_eval);
  eoStochTournamentSelect <Route> select_one;
  eoSelectNumber <Route> select (select_one, POP_SIZE);
  eoSGATransform <Route> transform (order_cross, CROSS_RATE, city_swap_mut, MUT_RATE);
  eoEPReplacement <Route> replace (2);
  eoEasyEA< Route > eaAlg( checkpoint, eval, select, transform, replace );
  peoWrapper parallelEA( eaAlg, pop);
  peo :: run ();
  peo :: finalize (); 

  if (getNodeRank()==1)
  {
  	pop.sort();
    std :: cout << "\nResult before the local search\n";
    for(unsigned i=0;i<pop.size();i++)
    	std::cout<<"\n"<<pop[i].fitness();
  }
  
// Local search
  peo :: init (__argc, __argv);
  peoMultiStart <Route> initParallelHC (hc);
  peoWrapper parallelHC (initParallelHC, pop);
  initParallelHC.setOwner(parallelHC);
  peo :: run( );
  peo :: finalize( );

  if (getNodeRank()==1)
  {
    std :: cout << "\nResult after the local search\n";
    pop.sort();
    for(unsigned i=0;i<pop.size();i++)
    	std::cout<<"\n"<<pop[i].fitness();
  }
}
