/*
* <moRandImprSelect.h>
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

#ifndef __moRandImprSelect_h
#define __moRandImprSelect_h

#include <vector>

#include <utils/eoRNG.h>
#include "moMoveSelect.h"

//! One of the possible moMove selector (moMoveSelect)
/*!
  All the neighbors are considered.
  One of them that enables an improvment of the objective function is choosen.
*/
template < class M > class moRandImprSelect:public moMoveSelect < M >
  {

  public:

    //! Alias for the fitness
    typedef typename M::EOType::Fitness Fitness;

    //!Procedure which all that needs a moRandImprSelect
    /*!
       Give a value to the initialise fitness.
       Clean the move and fitness vectors.

       \param __fit the current best fitness
     */
    void init (const Fitness & __fit)
    {
      init_fit = __fit;
      vect_better_fit.clear ();
      vect_better_moves.clear ();
    }

    //! Function that updates the fitness and move vectors
    /*!
       if a move give a better fitness than the initial fitness, 
       it is saved and the fitness too.

       \param __move a new move.
       \param __fit a new fitness associated to the new move.
       \return TRUE.
     */
    bool update (const M & __move, const Fitness & __fit)
    {

      if (__fit > init_fit)
        {

          vect_better_fit.push_back (__fit);
          vect_better_moves.push_back (__move);
        }

      return true;
    }

    //! The move selection
    /*!
       One the saved move is randomly chosen.

       \param __move the reference of the move that can be initialised by the function.
       \param __fit the reference of the fitness that can be initialised by the function.
       \throws EmptySelection If no move which improves the current fitness are found.
     */
    void operator   () (M & __move, Fitness & __fit) throw (EmptySelection)
    {

      if (!vect_better_fit.empty ())
        {

          unsigned n = rng.random (vect_better_fit.size ());

          __move = vect_better_moves[n];
          __fit = vect_better_fit[n];
        }
      else
        throw EmptySelection ();
    }

  private:

    //! Fitness of the current solution.
    Fitness init_fit;

    //! Candidate fitnesse vector.
    std::vector < Fitness > vect_better_fit;

    //! Candidate move vector.
    std::vector < M > vect_better_moves;
  };

#endif
