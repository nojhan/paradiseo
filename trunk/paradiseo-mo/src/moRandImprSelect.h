/*
  <moRandImprSelect.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
  (C) OPAC Team, LIFL, 2002-2008
 
  Sébastien Cahon, Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)
 
  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".
 
  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited liability.
 
  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.
 
  ParadisEO WebSite : http://paradiseo.gforge.inria.fr
  Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef _moRandImprSelect_h
#define _moRandImprSelect_h

#include <vector>
#include <utils/eoRNG.h>
#include <moMoveSelect.h>

//! One of the possible moMove selector (moMoveSelect)
/*!
  All the neighbors are considered.
  One of them that enables an improvment of the objective function is choosen.
*/
template < class M > 
class moRandImprSelect: public moMoveSelect < M >
{
 public:

  //! Alias for the fitness
  typedef typename M::EOType::Fitness Fitness;

  //!Procedure which all that needs a moRandImprSelect
  /*!
    Give a value to the initialise fitness.
    Clean the move and fitness vectors.

    \param _fitness the current best fitness
  */
  void init (const Fitness & _fitness)
  {
    initial_fitness = _fitness;
    better_fitnesses.clear();
    better_moves.clear();
    firstTime=true;
  }

  //! Function that updates the fitness and move vectors
  /*!
    if a move give a better fitness than the initial fitness, 
    it is saved and the fitness too.

    \param _move a new move.
    \param _fitness a new fitness associated to the new move.
    \return true.
  */
  bool update (const M & _move, const Fitness & _fitness)
  {
    firstTime=false;

    if (_fitness > initial_fitness)
      {
	better_fitnesses.push_back(_fitness);
	better_moves.push_back(_move);
      }
  }

  //! The move selection
  /*!
    One the saved move is randomly chosen.

    \param _move the reference of the move that can be initialised by the function.
    \param _fitness the reference of the fitness that can be initialised by the function.
  */
  void operator () (M & _move, Fitness & _fitness)
  {
    unsigned int index;
    
    index=0;
    
    if( (better_fitnesses.size()==0) || (better_moves.size()==0) )
      {
	if(firstTime)
	  {
	    throw std::runtime_error("[moRandImprSelect.h]: no move or/and no fitness already saved, update has to be called first.");
	  }
	return;
      }

    index = rng.random (better_fitnesses.size ());
    
    _move = better_moves[index];
    _fitness = better_fitnesses[index];
  }

 private:

  //! Fitness of the current solution.
  Fitness initial_fitness;

  //! Candidate fitnesse vector.
  std::vector < Fitness > better_fitnesses;

  //! Candidate move vector.
  std::vector < M > better_moves;

  //! Indicate if update has been called or not.
  bool firstTime;
};

#endif
