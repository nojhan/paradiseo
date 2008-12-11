/*
  <moTA.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2008-2009
  (C) OPAC Team, LIFL, 2002-2008

  Salma Mesmoudi

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

#ifndef _moTA_h
#define _moTA_h

#include <math.h>

#include <eoEvalFunc.h>
#include <moAlgo.h>
#include <moRandMove.h>
#include <moMoveIncrEval.h>
#include <moCoolingSchedule.h>
#include <moSolContinue.h>

//! Threeshol Accepting (TA)
/*!
  Class that describes a Threeshol Accepting algorithm.
*/
template < class M >
class moTA:public moAlgo < typename M::EOType >
{
  //! Alias for the type
  typedef typename M::EOType EOT;

  //! Alias for the fitness
  typedef typename EOT::Fitness Fitness;


 public:

  //! TA constructor
  /*!
    All the boxes used by a SA need to be given.

    \param _random_move_generator The move generator (generally randomly).
    \param _incremental_evaluation The (generally) efficient evaluation function
    \param _continue The stopping criterion.
    \param _initial_threeshold The initial threeshold.
    \param _cooling_schedule The cooling schedule, describes how the temperature is modified.
    \param _full_evaluation The full evaluation function.
  */
  moTA (moRandMove < M > & _random_move_generator, moMoveIncrEval < M > & _incremental_evaluation,
	moSolContinue < EOT > & _continue, double _initial_threeshold, moCoolingSchedule & _cooling_schedule,
	eoEvalFunc < EOT > & _full_evaluation):
  random_move_generator(_random_move_generator), incremental_evaluation(_incremental_evaluation),
    continu(_continue), initial_threeshold(_initial_threeshold),
    cooling_schedule(_cooling_schedule), full_evaluation(_full_evaluation)
  {}

  //! function that launches the TA algorithm.
  /*!
    As a moSA the TA can be used for HYBRIDATION in an evolutionary algorithm.

    \param _solution A solution to improve.
    \return TRUE.
  */
  bool operator ()(EOT & _solution)
  {
    Fitness incremental_fitness, delta_fit;
    EOT best_solution;
    double threeshold;
    M move;

    if (_solution.invalid())
      {
	full_evaluation (_solution);
      }

    threeshold = initial_threeshold;

    best_solution = _solution;

    do
      {
	continu.init ();

	do
	  {
	    random_move_generator(move);

	    incremental_fitness = incremental_evaluation (move, _solution);

	    delta_fit = incremental_fitness - _solution.fitness ();

	    if ( threeshold > delta_fit)

	      {
		move(_solution);
		_solution.fitness(incremental_fitness);

		// Updating the best solution found  until now ?
		if ( _solution.fitness() > best_solution.fitness() )
		  {
		    best_solution = _solution;
		  }
	      }

	  }
	while ( continu (_solution));
      }
    while ( cooling_schedule (threeshold) );

    _solution = best_solution;

    return true;
  }

 private:

  //! A move generator (generally randomly)
  moRandMove < M > & random_move_generator;

  //! A (generally) efficient evaluation function.
  moMoveIncrEval < M > & incremental_evaluation;

  //! Stopping criterion before threeshold update
  moSolContinue < EOT > & continu;

  //! Initial temperature
  double  initial_threeshold;

  //! The cooling schedule
  moCoolingSchedule & cooling_schedule;

  //! A full evaluation function.
  eoEvalFunc < EOT > & full_evaluation;
};

#endif
