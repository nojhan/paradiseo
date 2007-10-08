/* 
* <moTS.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sébastien Cahon, Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)
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

#ifndef __moTS_h
#define __moTS_h

#include <eoOp.h>
#include <eoEvalFunc.h>

#include "moAlgo.h"
#include "moSolContinue.h"

#include "moMoveExpl.h"
#include "moTSMoveLoopExpl.h"


//! Tabu Search (TS)
/*!
  Generic algorithm that describes a tabu search.
 */
template < class M > class moTS:public moAlgo < typename M::EOType >
{

  //!Alias for the type
  typedef
    typename
    M::EOType
    EOT;

  //!Alias for the fitness
  typedef
    typename
    EOT::Fitness
    Fitness;

public:

  //!Constructor of a moTS specifying all the boxes
  /*!
     In this constructor, a moTSMoveLoopExpl is instanciated.

     \param __move_init move initialisation
     \param __next_move neighborhood explorer
     \param __incr_eval efficient evaluation
     \param __tabu_list tabu list
     \param __aspir_crit aspiration criterion
     \param __cont stop criterion
     \param __full_eval full evaluation function
   */
moTS (moMoveInit < M > &__move_init, moNextMove < M > &__next_move, moMoveIncrEval < M > &__incr_eval, moTabuList < M > &__tabu_list, moAspirCrit < M > &__aspir_crit, moSolContinue < EOT > &__cont, eoEvalFunc < EOT > &__full_eval):move_expl (*new moTSMoveLoopExpl < M >
	     (__move_init, __next_move, __incr_eval, __tabu_list,
	      __aspir_crit)), cont (__cont), full_eval (__full_eval)
      {}

  //! Constructor with less parameters
  /*!
     The explorer is given in the parameters.

     \param __move_expl the explorer (generally different that a moTSMoveLoopExpl)
     \param __cont stop criterion
     \param __full_eval full evaluation function
   */
moTS (moMoveExpl < M > &__move_expl, moSolContinue < EOT > &__cont, eoEvalFunc < EOT > &__full_eval):move_expl (__move_expl),
    cont (__cont),
    full_eval (__full_eval)
    {}

  //! Function which launchs the Tabu Search
  /*!
     Algorithm of the tabu search.
     As a moSA or a moHC, it can be used for HYBRIDATION in an evolutionary algorithm.
     For security a lock (pthread_mutex_t) is closed during the algorithm. 

     \param __sol a solution to improve.
     \return TRUE.
   */
  bool operator   ()(EOT & __sol)
  {
    if (__sol.invalid ())
      {
	full_eval (__sol);
      }

    M move;

    EOT best_sol = __sol, new_sol;

    cont.init ();

    do
      {

	new_sol = __sol;

	try
	{

	  move_expl (__sol, new_sol);

	}
	catch (EmptySelection & __ex)
	{

	  break;
	}

	/* Updating the best solution
	   found until now ? */
	if (new_sol.fitness () > __sol.fitness ())
	  {
	    best_sol = new_sol;
	  }

	__sol = new_sol;

      }
    while (cont (__sol));

    __sol = best_sol;
 
    return true;
  }

private:

  //! Neighborhood explorer
  moMoveExpl < M > &move_expl;

  //! Stop criterion
  moSolContinue < EOT > &cont;

  //! Full evaluation function
  eoEvalFunc < EOT > &full_eval;
};

#endif
