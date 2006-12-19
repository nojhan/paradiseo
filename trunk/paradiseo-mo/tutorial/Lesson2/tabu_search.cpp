// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "tabu_search.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <moTS.h>
#include <moNoAspirCrit.h>
#include <moImprBestFitAspirCrit.h>
#include <moGenSolContinue.h>
#include <moFirstImprSelect.h>
#include <moBestImprSelect.h>
#include <moRandImprSelect.h>

#include <graph.h>
#include <route.h>
#include <route_eval.h>
#include <route_init.h>

#include <two_opt.h>
#include <two_opt_init.h>
#include <two_opt_next.h>
#include <two_opt_incr_eval.h>
#include <two_opt_tabu_list.h>

int main (int __argc, char * __argv []) {

  if (__argc != 2) {
    
    std :: cerr << "Usage : ./tabu_search [instance]" << std :: endl ;
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

  TwoOptTabuList tabu_list ; // Tabu List

  moNoAspirCrit <TwoOpt> aspir_crit ; // Aspiration Criterion

  moGenSolContinue <Route> cont (50000) ; // Continuator

  moTS <TwoOpt> tabu_search (two_opt_init, two_opt_next, two_opt_incr_eval, tabu_list, aspir_crit, cont, full_eval) ;
  tabu_search (route) ;

  std :: cout << "[To] " << route << std :: endl ;

  return 0 ;
}

