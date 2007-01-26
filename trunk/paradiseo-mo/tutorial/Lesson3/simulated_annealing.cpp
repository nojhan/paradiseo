// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "simulated_annealing.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT

   Contact:  paradiseo-help@lists.gforge.inria.fr
*/

#include <moSA.h>
#include <moEasyCoolSched.h>
#include <moGenSolContinue.h>

#include <graph.h>
#include <route.h>
#include <route_eval.h>
#include <route_init.h>

#include <two_opt.h>
#include <two_opt_rand.h>
#include <two_opt_incr_eval.h>

int main (int __argc, char * __argv []) {

  if (__argc != 2) {
    
    std :: cerr << "Usage : ./simulated_annealing [instance]" << std :: endl ;
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
  
  TwoOptRand two_opt_rand ; // Route Random. Gen.
     
  TwoOptIncrEval two_opt_incr_eval ; // Eff. eval.
  
  TwoOpt move ;
  
  moEasyCoolSched cool_sched (0.1, 0.98) ; // Cooling Schedule 
  
  moGenSolContinue <Route> cont (1000) ; /* Temperature Descreasing
					    will occur each 1000
					    iterations */ 
  
  moSA <TwoOpt> simulated_annealing (two_opt_rand, two_opt_incr_eval, cont, 1000, cool_sched, full_eval) ;
  simulated_annealing (route) ;

  std :: cout << "[To] " << route << std :: endl ;
  
  return 0 ;
}

