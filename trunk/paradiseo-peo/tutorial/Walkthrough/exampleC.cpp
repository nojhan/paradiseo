/* 
* <exampleC.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sebastien Cahon, Alexandru-Adrian Tantar
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
// (c) OPAC Team, LIFL, July 2007
//
// Contact: paradiseo-help@lists.gforge.inria.fr

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
#define NUM_GEN 10
#define CROSS_RATE 1.0
#define MUT_RATE 0.01

#define MIG_FREQ 1 
#define MIG_SIZE 5


int main (int __argc, char * * __argv) {

  peo :: init (__argc, __argv);

  
  loadParameters (__argc, __argv); /* Processing some parameters relative to the tackled
				      problem (TSP) */

  /* Migration topology */
  RingTopology topo;



  // The First EA -------------------------------------------------------------------------------------
  
  RouteInit route_init; /* Its builds random routes */
  RouteEval full_eval; /* Full route evaluator */

  OrderXover order_cross; /* Recombination */
  CitySwap city_swap_mut;  /* Mutation */
  
  eoPop <Route> ox_pop (POP_SIZE, route_init);  /* Population */
  
  eoGenContinue <Route> ox_cont (NUM_GEN); /* A fixed number of iterations */  
  eoCheckPoint <Route> ox_checkpoint (ox_cont); /* Checkpoint */
  peoSeqPopEval <Route> ox_pop_eval (full_eval);  
  eoStochTournamentSelect <Route> ox_select_one;
  eoSelectNumber <Route> ox_select (ox_select_one, POP_SIZE);
  eoSGATransform <Route> ox_transform (order_cross, CROSS_RATE, city_swap_mut, MUT_RATE);
  peoSeqTransform <Route> ox_seq_transform (ox_transform);    
  eoEPReplacement <Route> ox_replace (2);

  
  /* The migration policy */
  eoPeriodicContinue <Route> ox_mig_cont (MIG_FREQ); /* Migration occurs periodically */
  eoStochTournamentSelect <Route> ox_mig_select_one; /* Emigrants are randomly selected */
  eoSelectNumber <Route> ox_mig_select (ox_mig_select_one, MIG_SIZE);
  eoPlusReplacement <Route> ox_mig_replace; /* Immigrants replace the worse individuals */
  
  peoAsyncIslandMig <Route> ox_mig (ox_mig_cont, ox_mig_select, ox_mig_replace, topo, ox_pop, ox_pop);
  ox_checkpoint.add (ox_mig);
  
  peoEA <Route> ox_ea (ox_checkpoint, ox_pop_eval, ox_select, ox_seq_transform, ox_replace);
  ox_mig.setOwner (ox_ea);
  
  ox_ea (ox_pop);   /* Application to the given population */
  // --------------------------------------------------------------------------------------------------
  


  // The Second EA ------------------------------------------------------------------------------------

  RouteInit route_init2; /* Its builds random routes */
  RouteEval full_eval2; /* Full route evaluator */

  OrderXover order_cross2; /* Recombination */
  CitySwap city_swap_mut2;  /* Mutation */


  eoPop <Route> ox_pop2 (POP_SIZE, route_init2);  /* Population */


  eoGenContinue <Route> ox_cont2 (NUM_GEN); /* A fixed number of iterations */
  eoCheckPoint <Route> ox_checkpoint2 (ox_cont2); /* Checkpoint */
  peoSeqPopEval <Route> ox_pop_eval2 (full_eval2);
  eoStochTournamentSelect <Route> ox_select_one2;
  eoSelectNumber <Route> ox_select2 (ox_select_one2, POP_SIZE);
  eoSGATransform <Route> ox_transform2 (order_cross2, CROSS_RATE, city_swap_mut2, MUT_RATE);
  peoSeqTransform <Route> ox_seq_transform2 (ox_transform2);
  eoEPReplacement <Route> ox_replace2 (2);

  /* The migration policy */
  eoPeriodicContinue <Route> ox_mig_cont2 (MIG_FREQ); /* Migration occurs periodically */
  eoStochTournamentSelect <Route> ox_mig_select_one2; /* Emigrants are randomly selected */
  eoSelectNumber <Route> ox_mig_select2 (ox_mig_select_one2, MIG_SIZE);
  eoPlusReplacement <Route> ox_mig_replace2; /* Immigrants replace the worse individuals */

  peoAsyncIslandMig <Route> ox_mig2 (ox_mig_cont2, ox_mig_select2, ox_mig_replace2, topo, ox_pop2, ox_pop2);
  ox_checkpoint2.add (ox_mig2);

  peoEA <Route> ox_ea2 (ox_checkpoint2, ox_pop_eval2, ox_select2, ox_seq_transform2, ox_replace2);
  ox_mig2.setOwner (ox_ea2);

  ox_ea2 (ox_pop2);   /* Application to the given population */
  // --------------------------------------------------------------------------------------------------



  peo :: run ();
  peo :: finalize (); /* Termination */


  // rank 0 is assigned to the scheduler in the XML mapping file
  if ( getNodeRank() == 1 ) { 

    std::cout << "EA[ 0 ] -----> " << ox_pop.best_element().fitness() << std::endl;
    std::cout << "EA[ 1 ] -----> " << ox_pop2.best_element().fitness() << std::endl;
  }


  return 0;
}
