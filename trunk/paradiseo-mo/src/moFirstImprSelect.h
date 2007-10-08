/* 
* <moFirstImprSelect.h>
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

#ifndef __moFirstImprSelect_h
#define __moFirstImprSelect_h

#include "moMoveSelect.h"

//! One possible moMoveSelect.
/*!
  The neighborhood is explored until
  a move enables an improvment of the
  current solution.
*/
template < class M > class moFirstImprSelect:public moMoveSelect < M >
{

public:

  //! Alias for the fitness.
  typedef typename M::EOType::Fitness Fitness;

  //! Procedure which initialise the exploration.
  /*!
     It save the current fitness as the initial value for the fitness.
   */
  virtual void init (const Fitness & __fit)
  {

    valid = false;
    init_fit = __fit;
  }


  //!Function that indicates if the current move has not improved the fitness.
  /*!
     If the given fitness enables an improvment,
     the move (moMove) should be applied to the current solution.

     \param __move a move.
     \param __fit a fitness linked to the move.
     \return TRUE if the move does not improve the fitness.
   */
  bool update (const M & __move, const typename M::EOType::Fitness & __fit)
  {

    if (__fit > init_fit)
      {

	best_fit = __fit;
	best_move = __move;
	valid = true;

	return false;
      }
    else
      {
	return true;
      }
  }

  //! Procedure which saved the best move and fitness.
  /*!
     \param __move the current move (result of the procedure).
     \param __fit the current fitness (result of the procedure).
     \throws EmptySelection if no move has improved the fitness.
   */
  void operator   () (M & __move, Fitness & __fit) throw (EmptySelection)
  {

    if (valid)
      {
	__move = best_move;
	__fit = best_fit;
      }
    else
      throw EmptySelection ();
  }

private:

  //! Allow to know if at least one move has improved the solution.
  bool valid;

  //! Best stored movement.
  M best_move;

  //! Initial fitness.
  Fitness init_fit;

  //! Best stored fitness.
  Fitness best_fit;

};

#endif
