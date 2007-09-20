// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "iterated_local_search.cpp"

// (c) OPAC Team, LIFL, 2003-2007

/* LICENCE TEXT

   Contact:  paradiseo-help@lists.gforge.inria.fr
*/

#include <mo.h>

#include <graph.h>
#include <route.h>
#include <route_eval.h>
#include <route_init.h>

#include <two_opt.h>
#include <two_opt_init.h>
#include <two_opt_next.h>
#include <two_opt_incr_eval.h>
#include <two_opt_tabu_list.h>
#include <two_opt_rand.h>

#include <city_swap.h>

int
main (int __argc, char * __argv []) 
{
  if (__argc != 2) 
    {
      std :: cerr << "Usage : ./iterated_local_search [instance]" << std :: endl ;
      return 1 ;
    }
  
  Graph :: load (__argv [1]) ; // Instance

  Route route ; // Solution
  
  RouteInit init ; // Sol. Random Init.
  init (route) ;

  RouteEval full_eval ; // Full. Eval.
  full_eval (route) ;
  
  std :: cout << "[From] " << route << std :: endl ;
  
  /* Tools for an efficient (? :-))
     local search ! */

  TwoOptInit two_opt_init ; // Init.
   
  TwoOptNext two_opt_next ; // Explorer.
  
  TwoOptIncrEval two_opt_incr_eval ; // Eff. eval.
  
  moBestImprSelect <TwoOpt> two_opt_select ;
  
  //moHC<TwoOpt> hill_climbing (two_opt_init, two_opt_next, two_opt_incr_eval, two_opt_select, full_eval) ;
    
  moGenSolContinue <Route> cont (1000) ; 
  
  moFitComparator<Route> comparator;

  CitySwap perturbation;

  //moILS<TwoOpt> iterated_local_search (hill_climbing, cont, comparator, perturbation, full_eval) ;
  //moILS<TwoOpt> iterated_local_search (two_opt_init, two_opt_next, two_opt_incr_eval, 
  //			       two_opt_select, cont, comparator, perturbation, full_eval) ;
  
  moGenSolContinue <Route> ts_cont (100) ; 

  TwoOptTabuList tabu_list ; // Tabu List
  moNoAspirCrit <TwoOpt> aspir_crit ; // Aspiration Criterion

  //moILS<TwoOpt> iterated_local_search (two_opt_init, two_opt_next, two_opt_incr_eval, tabu_list, aspir_crit, ts_cont, 
  //				       cont, comparator, perturbation, full_eval) ;
  
  TwoOptRand two_opt_rand ; // Route Random. Gen.

  moExponentialCoolingSchedule cool_sched (0.1, 0.98) ; // Exponential Cooling Schedule 
  
  moGenSolContinue <Route> sa_cont (100) ; 

  moILS<TwoOpt> iterated_local_search (two_opt_rand, two_opt_incr_eval, sa_cont, 100, cool_sched,
				       cont, comparator, perturbation, full_eval) ;

  iterated_local_search(route) ;

  std :: cout << "[To] " << route << std :: endl ;
  
  return 0 ;
}

