// "main_ga.cpp"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
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

#include <paradiseo>

#define POP_SIZE 10
#define NUM_GEN 100
#define CROSS_RATE 1.0
#define MUT_RATE 0.01
#define NUM_PART_EVALS 2

#define MIG_FREQ 10
#define MIG_SIZE 10
#define HYBRID_SIZE 3

int main (int __argc, char * * __argv) {

  peo :: init (__argc, __argv);

  loadParameters (__argc, __argv); /* Processing some parameters relative to the tackled
				      problem (TSP) */

  RouteInit route_init; /* Its builds random routes */  
  RouteEval full_eval; /* Full route evaluator */

  MergeRouteEval merge_eval; 
  
  std :: vector <eoEvalFunc <Route> *> part_eval;
  for (unsigned i = 1 ; i <= NUM_PART_EVALS ; i ++)
    part_eval.push_back (new PartRouteEval ((float) (i - 1) / NUM_PART_EVALS, (float) i / NUM_PART_EVALS));
  
  OrderXover order_cross; /* Recombination */
  PartialMappedXover pm_cross;
  EdgeXover edge_cross;
  CitySwap city_swap_mut;  /* Mutation */

  RingTopology topo;
 
  /** The first EA **/

  eoPop <Route> ox_pop (POP_SIZE, route_init);  /* Population */
  
  eoGenContinue <Route> ox_cont (NUM_GEN); /* A fixed number of iterations */  
  eoCheckPoint <Route> ox_checkpoint (ox_cont); /* Checkpoint */
  peoParaPopEval <Route> ox_pop_eval (part_eval, merge_eval);  
  eoStochTournamentSelect <Route> ox_select_one;
  eoSelectNumber <Route> ox_select (ox_select_one, POP_SIZE);
  eoSGATransform <Route> ox_transform (order_cross, CROSS_RATE, city_swap_mut, MUT_RATE);
  peoSeqTransform <Route> ox_para_transform (ox_transform);    
  eoEPReplacement <Route> ox_replace (2);
  
  /* The migration policy */
  eoPeriodicContinue <Route> ox_mig_cont (MIG_FREQ); /* Migration occurs periodically */
  eoRandomSelect <Route> ox_mig_select_one; /* Emigrants are randomly selected */
  eoSelectNumber <Route> ox_mig_select (ox_mig_select_one, MIG_SIZE);
  eoPlusReplacement <Route> ox_mig_replace; /* Immigrants replace the worse individuals */
  
  peoAsyncIslandMig <Route> ox_mig (ox_mig_cont, ox_mig_select, ox_mig_replace, topo, ox_pop, ox_pop);
  //peoSyncIslandMig <Route> ox_mig (MIG_FREQ, ox_mig_select, ox_mig_replace, topo, ox_pop, ox_pop);
  
  ox_checkpoint.add (ox_mig);
  
  peoEA <Route> ox_ea (ox_checkpoint, ox_pop_eval, ox_select, ox_para_transform, ox_replace);
  ox_mig.setOwner (ox_ea);
  
  ox_ea (ox_pop);   /* Application to the given population */    

  /** The second EA **/

  eoPop <Route> pmx_pop (POP_SIZE, route_init);  /* Population */

  eoGenContinue <Route> pmx_cont (NUM_GEN); /* A fixed number of iterations */  
  eoCheckPoint <Route> pmx_checkpoint (pmx_cont); /* Checkpoint */
  peoSeqPopEval <Route> pmx_pop_eval (full_eval);  
  eoRankingSelect <Route> pmx_select_one;
  eoSelectNumber <Route> pmx_select (pmx_select_one, POP_SIZE);
  eoSGATransform <Route> pmx_transform (pm_cross, CROSS_RATE, city_swap_mut, MUT_RATE);
  peoSeqTransform <Route> pmx_para_transform (pmx_transform);    
  eoPlusReplacement <Route> pmx_replace;

  /* The migration policy */
  eoPeriodicContinue <Route> pmx_mig_cont (MIG_FREQ); /* Migration occurs periodically */
  eoRandomSelect <Route> pmx_mig_select_one; /* Emigrants are randomly selected */
  eoSelectNumber <Route> pmx_mig_select (pmx_mig_select_one, MIG_SIZE);
  eoPlusReplacement <Route> pmx_mig_replace; /* Immigrants replace the worse individuals */
  peoAsyncIslandMig <Route> pmx_mig (pmx_mig_cont, pmx_mig_select, pmx_mig_replace, topo, pmx_pop, pmx_pop);
  //peoSyncIslandMig <Route> pmx_mig (MIG_FREQ, pmx_mig_select, pmx_mig_replace, topo, pmx_pop, pmx_pop);
  pmx_checkpoint.add (pmx_mig);
  
  /* Hybridization with a Local Search */
  TwoOptInit pmx_two_opt_init;
  TwoOptNext pmx_two_opt_next;
  TwoOptIncrEval pmx_two_opt_incr_eval;
  moBestImprSelect <TwoOpt> pmx_two_opt_move_select;
  moHC <TwoOpt> hc (pmx_two_opt_init, pmx_two_opt_next, pmx_two_opt_incr_eval, pmx_two_opt_move_select, full_eval);

  eoPeriodicContinue <Route> pmx_ls_cont (MIG_FREQ); /* Hybridization occurs periodically */
  eoRandomSelect <Route> pmx_ls_select_one; /* ? */
  eoSelectNumber <Route> pmx_ls_select (pmx_ls_select_one, HYBRID_SIZE); 
  eoPlusReplacement <Route> pmx_ls_replace;

  peoSyncMultiStart <Route> pmx_ls (pmx_ls_cont, pmx_ls_select, pmx_ls_replace, hc, pmx_pop);
  pmx_checkpoint.add (pmx_ls);

  peoEA <Route> pmx_ea (pmx_checkpoint, pmx_pop_eval, pmx_select, pmx_para_transform, pmx_replace);
  pmx_mig.setOwner (pmx_ea);
  pmx_ls.setOwner (pmx_ea);

  pmx_ea (pmx_pop);   /* Application to the given population */    

  /** The third EA **/

  eoPop <Route> edge_pop (POP_SIZE, route_init);  /* Population */

  eoGenContinue <Route> edge_cont (NUM_GEN); /* A fixed number of iterations */  
  eoCheckPoint <Route> edge_checkpoint (edge_cont); /* Checkpoint */
  peoSeqPopEval <Route> edge_pop_eval (full_eval);  
  eoRankingSelect <Route> edge_select_one;
  eoSelectNumber <Route> edge_select (edge_select_one, POP_SIZE); 	 
  peoParaSGATransform <Route> edge_para_transform (edge_cross, CROSS_RATE, city_swap_mut, MUT_RATE);
  eoPlusReplacement <Route> edge_replace;

  /* The migration policy */
  eoPeriodicContinue <Route> edge_mig_cont (MIG_FREQ); /* Migration occurs periodically */
  eoRandomSelect <Route> edge_mig_select_one; /* Emigrants are randomly selected */
  eoSelectNumber <Route> edge_mig_select (edge_mig_select_one, MIG_SIZE);
  eoPlusReplacement <Route> edge_mig_replace; /* Immigrants replace the worse individuals */
  peoAsyncIslandMig <Route> edge_mig (edge_mig_cont, edge_mig_select, edge_mig_replace, topo, edge_pop, edge_pop);
  //peoSyncIslandMig <Route> edge_mig (MIG_FREQ, edge_mig_select, edge_mig_replace, topo, edge_pop, edge_pop);
  edge_checkpoint.add (edge_mig);

  peoEA <Route> edge_ea (edge_checkpoint, edge_pop_eval, edge_select, edge_para_transform, edge_replace);

  edge_mig.setOwner (edge_ea);

  edge_ea (edge_pop);   /* Application to the given population */    
  
  peo :: run ();

  peo :: finalize (); /* Termination */

  return 0;
}
