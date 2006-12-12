// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "gen_algo.cpp"

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

#include <eoEasyEA.h>
#include <eoGenContinue.h>
#include <eoStochTournamentSelect.h>
#include <eoSGATransform.h>
#include <eoSelectNumber.h>

#include "share/graph.h"
#include "share/route.h"
#include "share/route_init.h"
#include "share/route_eval.h"
#include "share/order_xover.h"
#include "share/partial_mapped_xover.h"
#include "share/city_swap.h"

int main (int __argc, char * __argv []) {
  
  if (__argc != 2) {
    
    std :: cerr << "Usage : ./gen_algo [instance]" << std :: endl ;
    return 1 ;
  }

  Graph :: load (__argv [1]) ; // Instance

  RouteInit init ; // Sol. Random Init.

  RouteEval full_eval ; // Full Evaluator
   
  eoPop <Route> pop (100, init) ; // Population
  apply <Route> (full_eval, pop) ;

  std :: cout << "[From] " << pop.best_element () << std :: endl ;
  
  eoGenContinue <Route> cont (1000) ; /* Continuator (A fixed number of
					 1000 iterations */
 
  eoStochTournamentSelect <Route> select_one ; // Selector

  eoSelectNumber <Route> select (select_one, 100) ;

  //  OrderXover cross ; // Order Crossover
  PartialMappedXover cross ;

  CitySwap mut ; // City Swap Mutator
  
  eoSGATransform <Route> transform (cross, 1, mut, 0.01) ; 
  
  eoElitism <Route> merge (1) ; // Use of Elistism
  
  eoStochTournamentTruncate <Route> reduce (0.7) ; // Stoch. Replacement
  
  eoEasyEA <Route> ea (cont, full_eval, select, transform, merge, reduce) ;
  
  ea (pop) ;
  
  std :: cout << "[To] " << pop.best_element () << std :: endl ;
    
  return 0 ;
}

