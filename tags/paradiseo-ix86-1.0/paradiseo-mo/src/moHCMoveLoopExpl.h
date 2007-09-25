/* <moHCMoveLoopExpl.h>  
 *
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * Sebastien CAHON
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
 */

#ifndef __moHCMoveLoopExpl_h
#define __moHCMoveLoopExpl_h

#include "moMoveLoopExpl.h"

#include "moMoveInit.h"
#include "moNextMove.h"
#include "moMoveIncrEval.h"
#include "moMoveSelect.h"

//! Iterative explorer used by a moHC.
template < class M > class moHCMoveLoopExpl:public moMoveLoopExpl < M >
{

  //! Alias for the type.
  typedef typename M::EOType EOT;

  //! Alias for the fitness.
  typedef typename M::EOType::Fitness Fitness;

public:

  //! Constructor.
  /*!
     All the boxes have to be specified.

     \param __move_init the move initialiser.
     \param __next_move the neighborhood explorer.
     \param __incr_eval (generally) efficient evaluation function.
     \param __move_select the move selector.
   */
moHCMoveLoopExpl (moMoveInit < M > &__move_init, moNextMove < M > &__next_move, moMoveIncrEval < M > &__incr_eval, moMoveSelect < M > &__move_select):

  move_init (__move_init),
    next_move (__next_move),
    incr_eval (__incr_eval), move_select (__move_select)
  {

  }

  //!  Procedure which launches the explorer.
  /*!
     The exploration starts from an old solution and provides a new solution.

     \param __old_sol the current solution.
     \param __new_sol the new_sol (result of the procedure).
   */
  void operator   () (const EOT & __old_sol, EOT & __new_sol)
  {

    M move;

    //
    move_init (move, __old_sol);	/* Restarting the exploration of 
					   of the neighborhood ! */

    move_select.init (__old_sol.fitness ());

    while (move_select.update (move, incr_eval (move, __old_sol))
	   && next_move (move, __old_sol));

    try
    {

      M best_move;

      Fitness best_move_fit;

      move_select (best_move, best_move_fit);
      __new_sol.fitness (best_move_fit);
      best_move (__new_sol);

    }
    catch (EmptySelection & __ex)
    {

      // ?
    }
  }

private:

  //! Move initialiser.
  moMoveInit < M > &move_init;

  //! Neighborhood explorer.
  moNextMove < M > &next_move;

  //! (generally) Efficient evaluation.
  moMoveIncrEval < M > &incr_eval;

  //! Move selector.
  moMoveSelect < M > &move_select;

};

#endif
