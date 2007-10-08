/* 
* <tabu_search.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sébastien Cahon, Jean-Charles Boisson
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

int
main (int __argc, char * __argv []) 
{
  if (__argc != 2) 
    {
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
  //moSimpleMoveTabuList<TwoOpt> tabu_list(10);
  //moSimpleSolutionTabuList<TwoOpt> tabu_list(10);

  moNoAspirCrit <TwoOpt> aspir_crit ; // Aspiration Criterion

  moGenSolContinue <Route> cont (10000) ; // Continuator

  moTS <TwoOpt> tabu_search (two_opt_init, two_opt_next, two_opt_incr_eval, tabu_list, aspir_crit, cont, full_eval) ;
  tabu_search (route) ;
  
  std :: cout << "[To] " << route << std :: endl ;
  
  return 0 ;
}

