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
#define NUM_GEN 200
#define CROSS_RATE 1.0
#define MUT_RATE 0.1


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

// Local search
  TwoOptInit pmx_two_opt_init;
  TwoOptNext pmx_two_opt_next;
  TwoOptIncrEval pmx_two_opt_incr_eval;
  moBestImprSelect <TwoOpt> pmx_two_opt_move_select;
  moHC <TwoOpt> hc (pmx_two_opt_init, pmx_two_opt_next, pmx_two_opt_incr_eval, pmx_two_opt_move_select, full_eval);
  eoPop <Route> pop (POP_SIZE, route_init);
  peoMultiStart <Route> initParallelHC (hc);
  peoWrapper parallelHC (initParallelHC, pop);
  initParallelHC.setOwner(parallelHC);
  peo :: run( );
  peo :: finalize( );

  if (getNodeRank()==1)
  {
    std :: cout << "\nResult before the EA\n";
    pop.sort();
    for(unsigned i=0;i<pop.size();i++)
    	std::cout<<"\n"<<pop[i].fitness();
    std :: cout << "\n\n";
  }

// EA
  peo :: init (__argc, __argv);
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
    std :: cout << "\nResult after the EA\n";
    pop.sort();
    for(unsigned i=0;i<pop.size();i++)
    	std::cout<<"\n"<<pop[i].fitness();
    std :: cout << "\n\n";
  }
}
