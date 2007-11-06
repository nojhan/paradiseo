/* 
* <tsp.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sébastien Cahon, Thomas Legrand
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

#include <eoEasyEA.h>
#include <eoGenContinue.h>
#include <eoStochTournamentSelect.h>
#include <eoSGATransform.h>
#include <eoSelectNumber.h>

#include "graph.h"
#include "route.h"
#include "route_init.h"
#include "route_eval.h"
#include "order_xover.h"
#include "partial_mapped_xover.h"
#include "city_swap.h"

int main (int __argc, char * __argv []) {
  
  if (__argc != 2) {
    
    std :: cerr << "Usage : ./tsp [instance]" << std :: endl ;
    std :: cerr << "=> info: You can copy the benchs in the current directory by using 'make install'." << std :: endl ;
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

