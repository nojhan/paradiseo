// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "hill_climbing.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <moHC.h>
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



int main (int __argc, char * __argv []) {

  if (__argc != 2) {
    
    std :: cerr << "Usage : ./hill_climbing [instance]" << std :: endl ;
    return 1 ;
  }

  srand (1000) ;

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
  
  //moFirstImprSelect <TwoOpt> two_opt_select ;
  moBestImprSelect <TwoOpt> two_opt_select ;
  //moRandImprSelect <TwoOpt> two_opt_select ;

  moHC <TwoOpt> hill_climbing (two_opt_init, two_opt_next, two_opt_incr_eval, two_opt_select, full_eval) ;
  hill_climbing (route) ;

  std :: cout << "[To] " << route << std :: endl ;

  return 0 ;
}

