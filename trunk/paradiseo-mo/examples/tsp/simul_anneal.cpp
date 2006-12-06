// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "simul_anneal.cpp"

// (c) OPAC Team, LIFL, 2003

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: cahon@lifl.fr
*/

#include <moSA.h>
#include <moEasyCoolSched.h>
#include <moGenSolContinue.h>

#include "share/graph.h"
#include "share/route.h"
#include "share/route_eval.h"
#include "share/route_init.h"

#include "share/two_opt.h"
#include "share/two_opt_rand.h"
#include "share/two_opt_incr_eval.h"

int main (int __argc, char * __argv []) {

  if (__argc != 2) {
    
    std :: cerr << "Usage : ./simul_anneal [instance]" << std :: endl ;
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
					    will occur each 100
					    iterations */ 
  
  moSA <TwoOpt> simul_anneal (two_opt_rand, two_opt_incr_eval, cont, 1000, cool_sched, full_eval) ;
  simul_anneal (route) ;

  std :: cout << "[To] " << route << std :: endl ;
  
  return 0 ;
}

