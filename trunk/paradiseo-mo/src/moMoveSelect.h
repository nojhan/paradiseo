/*
* <moMoveSelect.h>
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

#ifndef __moMoveSelect_h
#define __moMoveSelect_h

#include <eoFunctor.h>

//! Special class that describes the case of no selection.
/*!
  This class is used as an exception that can be thrown if a solution selector has completly failed.
 */
class EmptySelection
  {};

//! Class that describes a move selector (moMove).
/*!
  It iteratively considers some moves (moMove) and their
  associated fitnesses. The best move is so regularly updated.
  At any time, it could be accessed.
*/
template < class M > class moMoveSelect:public eoBF < M &, typename M::EOType::Fitness &,
      void >
  {
  public:
    //! Alias for the fitness
    typedef
    typename
    M::EOType::Fitness
    Fitness;

    //! Procedure which initialises all that the move selector needs including the initial fitness.
    /*!
       In order to know the fitness of the solution,
       for which the neighborhood will
       be soon explored

       \param __fit the current fitness.
     */
    virtual void
    init (const Fitness & __fit) = 0;

    //! Function which updates the best solutions.
    /*!
       \param __move a new move.
       \param __fit a fitness linked to the new move.
       \return a boolean that expresses the need to resume the exploration.
     */
    virtual
    bool
    update (const M & __move, const Fitness & __fit) = 0;

  };

#endif
