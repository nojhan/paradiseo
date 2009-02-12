/*
  <moILS.h>
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

#ifndef _moILS_h
#define _moILS_h

#include <eoEvalFunc.h>

#include <moHC.h>
#include <moTS.h>
#include <moSA.h>

//! Iterated Local Search (ILS)
/*!
  Class which describes the algorithm for a iterated local search.
*/
template < class M >
class moILS:public moAlgo < typename M::EOType >
{
  //! Alias for the type.
  typedef typename M::EOType EOT;

  //! Alias for the fitness.
  typedef typename EOT::Fitness Fitness;

 public:

  //! Generic constructor
  /*!
    Generic constructor using a moAlgo

    \param _algorithm The solution based heuristic to use.
    \param _continue The stopping criterion.
    \param _acceptance_criterion The acceptance criterion.
    \param _perturbation The pertubation generator.
    \param _full_evaluation The evaluation function.
  */
  moILS (moAlgo<EOT> & _algorithm, moSolContinue <EOT> & _continue, moComparator<EOT> & _acceptance_criterion, 
	 eoMonOp<EOT> & _perturbation, eoEvalFunc<EOT> & _full_evaluation):
  algorithm(& _algorithm), continu(_continue), acceptance_criterion(_acceptance_criterion), 
    perturbation(_perturbation), full_evaluation(_full_evaluation), algorithm_memory_allocation(false)
  {}

  //! Constructor for using a moHC for the moAlgo
  /*!
    \param _move_initializer The move initialisation (for the moHC).
    \param _next_move_generator The move generator (for the moHC).
    \param _incremental_evaluation The partial evaluation function (for the moHC).
    \param _move_selection The move selection strategy (for the moHC).
    \param _continue The stopping criterion.
    \param _acceptance_criterion The acceptance criterion.
    \param _perturbation The pertubation generator.
    \param _full_evaluation The evaluation function.
  */
  moILS (moMoveInit < M > & _move_initializer, moNextMove < M > & _next_move_generator, 
	 moMoveIncrEval < M > & _incremental_evaluation, moMoveSelect < M > & _move_selection,
	 moSolContinue <EOT> & _continue, moComparator<EOT> & _acceptance_criterion,
	 eoMonOp<EOT> & _perturbation, eoEvalFunc<EOT> & _full_evaluation):
  algorithm(new moHC<M>(_move_initializer, _next_move_generator, _incremental_evaluation, _move_selection, _full_evaluation) ),
    continu(_continue), acceptance_criterion(_acceptance_criterion), perturbation(_perturbation), full_evaluation(_full_evaluation),
    algorithm_memory_allocation(true)
  {}

  //! Constructor for using a moTS for the moAlgo
  /*!
    \param _move_initializer The move initialisation (for the moTS).
    \param _next_move_generator The move generator (for the moTS).
    \param _incremental_evaluation The partial evaluation function (for the moTS).
    \param _tabu_list The tabu list (for the moTS !!!!).
    \param _aspiration_criterion The aspiration criterion (for the moTS).
    \param _moTS_continue The stopping criterion (for the moTS).
    \param _continue The stopping criterion.
    \param _acceptance_criterion The acceptance criterion.
    \param _perturbation The pertubation generator.
    \param _full_evaluation The evaluation function.
  */
  moILS (moMoveInit <M> & _move_initializer, moNextMove <M> & _next_move_generator, 
	 moMoveIncrEval <M> & _incremental_evaluation, moTabuList <M> & _tabu_list, 
	 moAspirCrit <M> & _aspiration_criterion, moSolContinue <EOT> & _moTS_continue,
	 moSolContinue <EOT> & _continue, moComparator<EOT> & _acceptance_criterion, eoMonOp<EOT> & _perturbation,
	 eoEvalFunc<EOT> & _full_evaluation):
  algorithm(new moTS<M>(_move_initializer, _next_move_generator, _incremental_evaluation, _tabu_list, _aspiration_criterion, 
		     _moTS_continue, _full_evaluation) ),
    continu(_continue), acceptance_criterion(_acceptance_criterion), perturbation(_perturbation), full_evaluation(_full_evaluation),
    algorithm_memory_allocation(true)
  {}
  
  //! Constructor for using a moSA for the moAlgo
  /*!
    \param _random_move_generator The random move generator (for the moSA).
    \param _incremental_evaluation The partial evaluation function (for the moSA).
    \param _moSA_continue The stopping criterion (for the moSA).
    \param _initial_temperature The initial temperature (for the moSA).
    \param _cooling_schedule The cooling schedule (for the moSA).
    \param _continue The stopping criterion.
    \param _acceptance_criterion The acceptance criterion.
    \param _perturbation The pertubation generator.
    \param _full_evaluation The evaluation function.
  */
  moILS (moRandMove<M> & _random_move_generator, moMoveIncrEval <M> & _incremental_evaluation, moSolContinue <EOT> & _moSA_continue, 
	 double _initial_temperature, moCoolingSchedule & _cooling_schedule, moSolContinue <EOT> & _continue, 
	 moComparator<EOT> & _acceptance_criterion, eoMonOp<EOT> & _perturbation, eoEvalFunc<EOT> & _full_evaluation):
  algorithm(new moSA<M>(_random_move_generator, _incremental_evaluation, _moSA_continue, _initial_temperature,
			_cooling_schedule, _full_evaluation) ),
    continu(_continue), acceptance_criterion(_acceptance_criterion), perturbation(_perturbation), full_evaluation(_full_evaluation),
    algorithm_memory_allocation(true)
      {}

  //! Destructor
  ~moILS()
    {
      if(algorithm_memory_allocation)
	{
	  delete(algorithm);
	}
    }

  //! Function which launches the ILS
  /*!
    The ILS has to improve a current solution.
    As the moSA, the moTS and the moHC, it can be used for HYBRIDATION in an evolutionnary algorithm.

    \param _solution a current solution to improve.
    \return true.
  */
  bool operator()(EOT & _solution)
  {
    EOT _solution_saved;
    
    if ( _solution.invalid() )
      {
	full_evaluation(_solution);
      } 
    
    _solution_saved=_solution;
    
    continu.init ();

    // some code has been duplicated in order to avoid one perturbation and one evaluation without adding a test in the loop.
    // better than a do {} while; with a test in the loop.
    
    (*algorithm)(_solution);

    if ( acceptance_criterion(_solution, _solution_saved) )
      {
	_solution_saved=_solution;

      }
    else
      {
	_solution=_solution_saved;
      }

    while ( continu (_solution) )
      {
	perturbation(_solution);
	full_evaluation(_solution);

	(*algorithm)(_solution);

	if ( acceptance_criterion(_solution, _solution_saved) )
	  {
	    _solution_saved=_solution;
	  }
	else
	  {
	    _solution=_solution_saved;
	  }
      }

    return true;
  }

 private:

  //! The solution based heuristic.
  moAlgo<EOT> * algorithm;

  //! The stopping criterion.
  moSolContinue<EOT> & continu;

  //! The acceptance criterion.
  moComparator<EOT> & acceptance_criterion;

  //! The perturbation generator
  eoMonOp<EOT> & perturbation;

  //! The full evaluation function
  eoEvalFunc<EOT> & full_evaluation;

  //! Indicate if the memory has been allocated for the algorithm.
  bool algorithm_memory_allocation;
};

#endif
