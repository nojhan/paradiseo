/* 
* <moTSMoveLoopExpl.h>
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

#ifndef __moTSMoveLoopExpl_h
#define __moTSMoveLoopExpl_h

#include "moMoveLoopExpl.h"

#include "moMoveInit.h"
#include "moNextMove.h"
#include "moMoveIncrEval.h"
#include "moMoveSelect.h"

#include "moTabuList.h"
#include "moAspirCrit.h"
#include "moBestImprSelect.h"

//! Explorer for a Tabu Search algorithm
/*!
  It is used by a moTS.
 */
template < class M > class moTSMoveLoopExpl:public moMoveLoopExpl < M >
{

  //!Alias for the type
  typedef typename M::EOType EOT;

  //!Alias for the fitness
  typedef typename M::EOType::Fitness Fitness;

public:

  //!Constructor
  /*!
     \param __move_init move initialisation
     \param __next_move neighborhood explorer
     \param __incr_eval efficient evaluation
     \param __tabu_list tabu list
     \param __aspir_crit aspiration criterion
   */
moTSMoveLoopExpl (moMoveInit < M > &__move_init, moNextMove < M > &__next_move, moMoveIncrEval < M > &__incr_eval, moTabuList < M > &__tabu_list, moAspirCrit < M > &__aspir_crit):
  move_init (__move_init),
    next_move (__next_move),
    incr_eval (__incr_eval),
    tabu_list (__tabu_list), aspir_crit (__aspir_crit)
  {

    tabu_list.init ();
    aspir_crit.init ();
  }

  //!Procedure which lauches the exploration
  /*!
     The exploration continues while the chosen move is not in the tabu list 
     or the aspiration criterion is true. If these 2 conditions are not true, the
     exploration stops if the move selector update function returns false.

     \param __old_sol the initial solution
     \param __new_sol the new solution
   */
  void operator   () (const EOT & __old_sol, EOT & __new_sol)
  {

    M move;


    move_init (move, __old_sol);	/* Restarting the exploration of 
					   of the neighborhood ! */

    move_select.init (__old_sol.fitness ());

    do
      {

	Fitness fit = incr_eval (move, __old_sol);

	if (!tabu_list (move, __old_sol) || aspir_crit (move, fit))
	  {
	    if (!move_select.update (move, fit))
	      break;
	  }

      }
    while (next_move (move, __old_sol));

    M best_move;

    Fitness best_move_fit;

    move_select (best_move, best_move_fit);

    __new_sol.fitness (best_move_fit);
    best_move (__new_sol);

    /* Removing moves that are
       no more tabu */
    tabu_list.update ();

    // Updating the tabu list
    tabu_list.add (best_move, __new_sol);
  }

private:

  //!Move initialisation
  moMoveInit < M > &move_init;

  //!Neighborhood explorer
  moNextMove < M > &next_move;

  //!Efficient evaluation
  moMoveIncrEval < M > &incr_eval;

  //!Move selector
  moBestImprSelect < M > move_select;

  //!Tabu list
  moTabuList < M > &tabu_list;

  //!Aspiration criterion
  moAspirCrit < M > &aspir_crit;
};

#endif
