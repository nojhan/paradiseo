/* <moHC.h>  
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

#ifndef __moHC_h
#define __moHC_h

#include <eoOp.h>
#include <eoEvalFunc.h>

#include "moAlgo.h"
#include "moMoveExpl.h"
#include "moHCMoveLoopExpl.h"

//! Hill Climbing (HC)
/*!
  Class which describes the algorithm for a hill climbing.
 */
template < class M > class moHC:public moAlgo < typename M::EOType >
{

  //! Alias for the type.
  typedef
    typename
    M::EOType
    EOT;

  //! Alias for the fitness.
  typedef
    typename
    EOT::Fitness
    Fitness;

public:

  //! Full constructor.
  /*!
     All the boxes are given in order the HC to use a moHCMoveLoopExpl.

     \param __move_init a move initialiser.
     \param __next_move a neighborhood explorer.
     \param __incr_eval a (generally) efficient evaluation function.
     \param __move_select a move selector.
     \param __full_eval a full evaluation function.
   */
moHC (moMoveInit < M > &__move_init, moNextMove < M > &__next_move, moMoveIncrEval < M > &__incr_eval, moMoveSelect < M > &__move_select, eoEvalFunc < EOT > &__full_eval):move_expl (*new moHCMoveLoopExpl < M >
	     (__move_init, __next_move, __incr_eval, __move_select)),
    full_eval (__full_eval)
  {

  }

  //! Light constructor.
  /*!
     This constructor allow to use another moMoveExpl (generally not a moHCMoveLoopExpl).

     \param __move_expl a complete explorer.
     \param __full_eval a full evaluation function.
   */
moHC (moMoveExpl < M > &__move_expl, eoEvalFunc < EOT > &__full_eval):move_expl (__move_expl),
    full_eval
    (__full_eval)
  {

  }

  //! Function which launches the HC
  /*!
     The HC has to improve a current solution.
     As the moSA and the mo TS, it can be used for HYBRIDATION in an evolutionnary algorithm.

     \param __sol a current solution to improve.
     \return TRUE.
   */
  bool operator   ()(EOT & __sol)
  {

    if (__sol.invalid ())
      {
	full_eval (__sol);
      }

    EOT new_sol;

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

	if (new_sol.fitness () > __sol.fitness ())
	  {
	    __sol = new_sol;
	  }
	else
	  {
	    break;
	  }

      }
    while (true);

    return true;
  }

private:

  //! Complete exploration of the neighborhood.
  moMoveExpl < M > &move_expl;

  //! A full evaluation function.
  eoEvalFunc < EOT > &full_eval;
};

#endif
