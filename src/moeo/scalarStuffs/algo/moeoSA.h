/*
   <moeoSA.h>
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

#ifndef __moeoSA_h
#define __moeoSA_h

#include "../../eo/eoEvalFunc.h"
#include "algo/moeoSolAlgo.h"
#include "moTSMoveLoopExpl.h"
#include "../fitness/moeoSingleObjectivization.h"
#include "explorer/moeoTSMoveLoopExpl.h"
#include "moSA.h"
//! Simulated annealing (SA)
/*!
  Generic algorithm that describes a Simulated Annealing algorithm.
  Adapts the moSA for a multi-objective problem using a moeoSingleObjectivization.
  M is for Move
 */

template < class M >
class moeoSA:public moeoSolAlgo < typename M::EOType >
{

	public:

		typedef typename M::EOType MOEOT;
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename MOEOT::Fitness Fitness;
		moeoSA (moRandMove < M > & _random_move_generator, moMoveIncrEval < M,ObjectiveVector > & _incremental_evaluation, 
				moSolContinue < MOEOT > & _continue, double _initial_temperature, moCoolingSchedule & _cooling_schedule,
				moeoSingleObjectivization<MOEOT> &_singler):
			incrEval(_singler,_incremental_evaluation),
			algo(_random_move_generator,incrEval,_continue,_initial_temperature,_cooling_schedule, _singler)
	{}
		/*!
		  Algorithm of the SA
		  As a moHC, it can be used for HYBRIDATION in an evolutionary algorithm.

		  \param _solution a solution to improve.
		  \return TRUE.
		 */
		bool operator()(MOEOT &_solution){
			return algo(_solution);
		}

	private:
		moeoIncrEvalSingleObjectivizer<MOEOT,M> incrEval;
		//! the actual algo
		moSA<M> algo;

};
#endif
