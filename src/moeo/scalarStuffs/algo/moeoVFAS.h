/*
   <moeoVFAS.h>
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

#ifndef _moeoVFAS_h
#define _moeoVFAS_h

#include <moComparator.h>

#include "../../eo/eoEvalFunc.h"
#include "algo/moeoSolAlgo.h"
#include "../fitness/moeoSingleObjectivization.h"
#include "../fitness/moeoAggregationFitnessAssignment.h"
#include "explorer/moeoHCMoveLoopExpl.h"
#include "weighting/moeoVariableWeightStrategy.h"
#include "weighting/moeoVariableRefPointStrategy.h"
#include "weighting/moeoDummyWeightStrategy.h"
#include "weighting/moeoDummyRefPointStrategy.h"
//! Variable fitness assignment search (vfas)
/*!
  Search using multiple fitness assignment to search solution to a multi objective problem
 */

template < class M >
class moeoVFAS:public moeoPopAlgo < typename M::EOType >
{

	public:
		typedef typename M::EOType MOEOT;
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename MOEOT::Fitness Fitness;
		/**
		 * constructor using a moAlgo and a vector of weight
		 * take a base vector of weight, and modify it to relaunch the algo with a diferent fitness
		 * use a selectOne to determine which moeot should be the base for the algo launch
		 * use a eoPop to keep result from each iteration
		 * @param _algorithm The solution based heuristic to use. It should at least use the fitness value at some point.
		 * @param _continue The stopping criterion.
		 * @param _select a selector to choose on which moeot we use the algorithm
		 * @param _weights a vector containing the base weights, which will be changed at each iteration.
		 * @param _eval The evaluation function.
		 * @param _wstrat the strategy to change weights (should be constructed with the same weights as the fitness)
		 */
		moeoVFAS (moeoSolAlgo<MOEOT> & _algorithm, eoContinue <MOEOT> & _continue,moeoSelectOne<MOEOT> &_select,
				std::vector<double> &_weights, eoEvalFunc< MOEOT > &_eval , moeoVariableWeightStrategy<MOEOT> &_wstrat ):
			algo(_algorithm),cont(_continue), select(_select), weights(_weights),eval(_eval),refPoint(defaultRefPoint), wstrat(_wstrat), rstrat(defaultRstrat)
	{}

		/**
		 * constructor using a moAlgo an ObjectiveVector and a vector of weight
		 * take a base vector of weight, and modify it to relaunch the algo with a diferent fitness
		 * use a selectOne to determine which moeot should be the base for the algo launch
		 * use a eoPop to keep result from each iteration
		 * @param _algorithm The solution based heuristic to use. It should at least use the fitness value at some point.
		 * @param _continue The stopping criterion.
		 * @param _select a selector to choose on which moeot we use the algorithm
		 * @param _weights a vector containing the base weights, which will be changed at each iteration.
		 * @param _refPoint a reference point changed at each iteration
		 * @param _eval The evaluation function.
		 * @param _wstrat the strategy to change weights (should be constructed with the same weights as the fitness)
		 * @param _rstrat the strategy to change the reference point
		 */
		moeoVFAS (moeoSolAlgo<MOEOT> & _algorithm, eoContinue <MOEOT> & _continue,moeoSelectOne<MOEOT> &_select,
				std::vector<double> &_weights, ObjectiveVector &_refPoint, eoEvalFunc< MOEOT > &_eval , moeoVariableWeightStrategy<MOEOT> &_wstrat , moeoVariableRefPointStrategy<MOEOT>& _rstrat):
			algo(_algorithm),cont(_continue), select(_select), weights(_weights),eval(_eval),refPoint(_refPoint),wstrat(_wstrat), rstrat(_rstrat)
	{}

		/**
		 * constructor without the weights
		 * @param _algorithm The solution based heuristic to use. It should at least use the fitness value at some point.
		 * @param _continue The stopping criterion.
		 * @param _select a selector to choose on which moeot we use the algorithm
		 * @param _eval The evaluation function.
		 * @param _wstrat the strategy to change weights (should be constructed with the same weights as the fitness)
		 */
		moeoVFAS (moeoSolAlgo<MOEOT> & _algorithm, eoContinue <MOEOT> & _continue,moeoSelectOne<MOEOT> &_select,
				eoEvalFunc< MOEOT > &_eval, moeoVariableWeightStrategy<MOEOT> &_wstrat):
			algo(_algorithm),cont(_continue), select(_select), weights(defaultWeights), eval(_eval), refPoint(defaultRefPoint), wstrat(defaultWstrat), rstrat(defaultRstrat)
	{
		weights.resize(MOEOT::ObjectiveVector::nObjectives(),1.0/MOEOT::ObjectiveVector::nObjectives());
	}
		/**
		 * launch the algorithm
		 * @param _pop the initial population on which algo will be launched
		 **/
		virtual void operator()(eoPop<MOEOT> &_pop){
			uniform_generator<double> rngGen(0.0,1.0);
			for (unsigned int i=0;i<_pop.size();i++){
				if (_pop[i].invalidObjectiveVector())
					eval(_pop[i]);
			}
			moeoObjectiveVectorNormalizer<MOEOT> norm(_pop);
			moeoAggregationFitnessAssignment<MOEOT> fitness(weights,eval);
			bool res=false;
			fitness(_pop);
			MOEOT moeot(select(_pop));
			wstrat(weights,moeot);
			rstrat(refPoint,moeot);

			do {
				norm.update_by_pop(_pop);
				fitness(_pop);
				moeot=(select(_pop));
				res=algo(moeot)||res;
				_pop.push_back(moeot);
				std::cout<<moeot.objectiveVector()<<std::endl;
				wstrat(weights,moeot);
				rstrat(refPoint,moeot);
			}while(cont(_pop));
		}

	private:
		moeoSolAlgo<MOEOT> &algo;
		eoContinue<MOEOT> &cont;
		moeoSelectOne<MOEOT> &select;
		std::vector<double> &weights;
		std::vector<double> defaultWeights;
		eoEvalFunc<MOEOT> &eval;
		ObjectiveVector &refPoint;
		ObjectiveVector defaultRefPoint;
		moeoVariableWeightStrategy<MOEOT> &wstrat;
		moeoVariableRefPointStrategy<MOEOT> &rstrat;
		moeoDummyRefPointStrategy<MOEOT> defaultRstrat;
		moeoDummyWeightStrategy<MOEOT> defaultWstrat;


};
#endif
