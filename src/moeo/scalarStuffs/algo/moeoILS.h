/*
   <moeoILS.h>
   Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
   (C) OPAC Team, LIFL, 2002-2008

   Sébastien Cahon, Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)
   François Legillon

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

#ifndef _moeoILS_h
#define _moeoILS_h

#include <moComparator.h>
#include <moILS.h>


#include "algo/moeoHC.h"
#include "algo/moeoTS.h"
#include "algo/moeoSA.h"
#include "../../eo/eoEvalFunc.h"
#include "algo/moeoSolAlgo.h"
#include "moHCMoveLoopExpl.h"
#include "../fitness/moeoSingleObjectivization.h"
#include "explorer/moeoHCMoveLoopExpl.h"

//! Iterated Local Search (ILS)
/*!
    Class which describes the algorithm for a iterated local search.
    Adapts the moILS for a multi-objective problem using a moeoSingleObjectivization.
    M is for Move
    */

template < class M >
class moeoILS:public moeoSolAlgo < typename M::EOType >
{

	public:
		typedef typename M::EOType MOEOT;
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename MOEOT::Fitness Fitness;
  //! Generic constructor
  /*!
    Generic constructor using a moAlgo

    \param _algorithm The solution based heuristic to use.
    \param _continue The stopping criterion.
    \param _acceptance_criterion The acceptance criterion.
    \param _perturbation The pertubation generator.
    \param _full_evaluation The evaluation function.
  */
		moeoILS (moeoSolAlgo<MOEOT> & _algorithm, moSolContinue <MOEOT> & _continue, moComparator<MOEOT> & _acceptance_criterion,
				eoMonOp<MOEOT> & _perturbation, moeoSingleObjectivization<MOEOT> & _full_evaluation):
			algo(_algorithm,_continue,_acceptance_criterion,_perturbation,_full_evaluation)
	{}
  //! Constructor for using a moHC
  /*!
    \param _move_initializer The move initialisation (for the moHC).
    \param _next_move_generator The move generator (for the moHC).
    \param _incremental_evaluation The partial evaluation function (for the moHC).
    \param _singler a singleObjectivizer to translate objectiveVectors into fitness
    \param _move_selection The move selection strategy (for the moHC).
    \param _continue The stopping criterion.
    \param _acceptance_criterion The acceptance criterion.
    \param _perturbation The pertubation generator.
  */
		moeoILS (moMoveInit < M > & _move_initializer, moNextMove < M > & _next_move_generator,
				moMoveIncrEval < M,ObjectiveVector > & _incremental_evaluation, moeoSingleObjectivization<MOEOT> &_singler,moMoveSelect < M > & _move_selection,
				moSolContinue <MOEOT> & _continue, moComparator<MOEOT> & _acceptance_criterion,
				eoMonOp<MOEOT> & _perturbation):
			subAlgo(new moeoHC<M>(_move_initializer,_next_move_generator,_incremental_evaluation,_move_selection,_singler)),
			algo(*subAlgo,_continue,_acceptance_criterion,_perturbation,_singler)
	{}
 //! Constructor for using a moTS for the moAlgo
  /*!
    \param _move_initializer The move initialisation (for the moTS).
    \param _next_move_generator The move generator (for the moTS).
    \param _incremental_evaluation The partial evaluation function (for the moTS).
    \param _singler a singleObjectivizer to translate objectiveVectors into fitness
    \param _tabu_list The tabu list (for the moTS !!!!).
    \param _aspiration_criterion The aspiration criterion (for the moTS).
    \param _moTS_continue The stopping criterion (for the moTS).
    \param _continue The stopping criterion.
    \param _acceptance_criterion The acceptance criterion.
    \param _perturbation The pertubation generator.
  */

		moeoILS (moMoveInit <M> & _move_initializer, moNextMove <M> & _next_move_generator,moMoveIncrEval <M,ObjectiveVector> & _incremental_evaluation, moeoSingleObjectivization<MOEOT> &_singler, moTabuList <M> & _tabu_list,moAspirCrit <M> & _aspiration_criterion, moSolContinue <MOEOT> & _moTS_continue,moSolContinue <MOEOT> & _continue, moComparator<MOEOT> & _acceptance_criterion, eoMonOp<MOEOT> & _perturbation):
			subAlgo(new moeoTS<M>(_move_initializer,_next_move_generator,_incremental_evaluation,_tabu_list,_aspiration_criterion,_moTS_continue,_singler)),
			algo((*subAlgo),_continue,_acceptance_criterion,_perturbation,_singler)
	{}
  //! Constructor for using a moSA for the moAlgo
  /*!
    \param _random_move_generator The random move generator (for the moSA).
    \param _incremental_evaluation The partial evaluation function (for the moSA).
    \param _singler a singleObjectivizer to translate objectiveVectors into fitness
    \param _moSA_continue The stopping criterion (for the moSA).
    \param _initial_temperature The initial temperature (for the moSA).
    \param _cooling_schedule The cooling schedule (for the moSA).
    \param _continue The stopping criterion.
    \param _acceptance_criterion The acceptance criterion.
    \param _perturbation The pertubation generator.
  */
		moeoILS (moRandMove<M> & _random_move_generator, moMoveIncrEval <M,ObjectiveVector> & _incremental_evaluation,moeoSingleObjectivization<MOEOT> &_singler, moSolContinue <MOEOT> & _moSA_continue,double _initial_temperature, moCoolingSchedule & _cooling_schedule, moSolContinue <MOEOT> & _continue,moComparator<MOEOT> & _acceptance_criterion, eoMonOp<MOEOT> & _perturbation):
			subAlgo(new moeoSA<M>(_random_move_generator, _incremental_evaluation, _moSA_continue, _initial_temperature,_cooling_schedule, _singler)),
			algo(*subAlgo,_continue, _acceptance_criterion, _perturbation, _singler)
	{}



		//! Function which launches the ILS
		/*!
		  The ILS has to improve a current solution.

		  \param _solution a current solution to improve.
		  \return true.
		 */
		bool operator()(MOEOT &_solution){
//			std::cout<<"moeoILS"<<std::endl;
			return algo(_solution);
		}

	private:
		moeoSolAlgo<MOEOT> *subAlgo;
		//! the actual algo
		moILS<M> algo;


};
#endif
