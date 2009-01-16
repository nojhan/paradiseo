/*
  <moTS.h>
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

#ifndef _moTS_h
#define _moTS_h

#include <eoEvalFunc.h>

#include <moAlgo.h>
#include <moSolContinue.h>
#include <moTSMoveLoopExpl.h>

//! Tabu Search (TS)
/*!
  Generic algorithm that describes a tabu search.
*/
template < class M >
class moTS:public moAlgo < typename M::EOType >
{
  //!Alias for the type
  typedef typename M::EOType EOT;

  //!Alias for the fitness
  typedef typename EOT::Fitness  Fitness;

 public:

  //!Constructor of a moTS specifying all the boxes
  /*!
    In this constructor, a moTSMoveLoopExpl is instanciated.

    \param _move_initializer The move initializer.
    \param _next_move_generator The neighbourhood explorer.
    \param _incremental_evaluation The (generally) efficient evaluation.
    \param _tabu_list The tabu list.
    \param _aspiration_criterion An aspiration criterion.
    \param _continue The stopping criterion.
    \param _full_evaluation A full evaluation function.
  */
  moTS (moMoveInit < M > & _move_initializer, moNextMove < M > & _next_move_generator,
	moMoveIncrEval < M > & _incremental_evaluation, moTabuList < M > & _tabu_list,
	moAspirCrit < M > & _aspiration_criterion, moSolContinue < EOT > & _continue,
	eoEvalFunc < EOT > & _full_evaluation):
  move_explorer (new moTSMoveLoopExpl < M >(_move_initializer, _next_move_generator, _incremental_evaluation,
					      _tabu_list,_aspiration_criterion) ),
    continu (_continue), full_evaluation (_full_evaluation), move_explorer_memory_allocation(true)
  {}

  //! Constructor with less parameters
  /*!
    The explorer is given in the parameters.

    \param _move_explorer The explorer (generally different that a moTSMoveLoopExpl).
    \param _continue The stopping criterion.
    \param _full_evaluation A full evaluation function.
  */
  moTS (moMoveExpl < M > & _move_explorer, moSolContinue < EOT > & _continue, eoEvalFunc < EOT > & _full_evaluation):
  move_explorer (&_move_explorer), continu (_continue), full_evaluation (_full_evaluation), move_explorer_memory_allocation(false)
  {}

  //! Destructor
  ~moTS()
    {
      if(move_explorer_memory_allocation)
	{
	  delete(move_explorer);
	}
    }

  //! Function which launchs the Tabu Search
  /*!
    Algorithm of the tabu search.
    As a moSA or a moHC, it can be used for HYBRIDATION in an evolutionary algorithm.
    For security a lock (pthread_mutex_t) is closed during the algorithm.

    \param _solution a solution to improve.
    \return TRUE.
  */
  bool operator ()(EOT & _solution)
  {
    M move;

    EOT best_solution, new_solution;

    if ( _solution.invalid () )
      {
	full_evaluation (_solution);
      }

    best_solution=_solution;

    // code used for avoiding warning because new_solution is indirectly initialized by move_expl.
    new_solution=_solution;

    continu.init ();

    do
      {
	(*move_explorer) (_solution, new_solution);

	// Updating the best solution found until now ?
	if (new_solution.fitness() > best_solution.fitness())
	  {
	    best_solution = new_solution;
	  }

	_solution = new_solution;
      }
    while ( continu (_solution) );

    _solution = best_solution;

    return true;
  }

 private:

  //! Neighborhood explorer
  moMoveExpl < M > * move_explorer;

  //! Stop criterion
  moSolContinue < EOT > & continu;

  //! Full evaluation function
  eoEvalFunc < EOT > & full_evaluation;

  //! Indicate if the memory has been allocated for the move_explorer.
  bool move_explorer_memory_allocation;
};

#endif
