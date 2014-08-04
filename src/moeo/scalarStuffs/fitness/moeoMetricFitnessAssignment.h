/*
 * <moeoMetricFitnessAssignment.h>
 * Copyright (C) DOLPHIN Project-Team, INRIA Lille-Nord Europe, 2006-2008
 * (C) OPAC Team, LIFL, 2002-2008
 *
 * François Legillon
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
//-----------------------------------------------------------------------------
// moeoMetrucFitnessAssignment.h
//-----------------------------------------------------------------------------
#ifndef MOEOMETRICFITNESSASSIGNMENT_H_
#define MOEOMETRICFITNESSASSIGNMENT_H_

//include <moeo>
#include <vector>
#include "../../../eo/eoPop.h"
#include "../../fitness/moeoSingleObjectivization.h"
#include "../../utils/moeoObjectiveVectorNormalizer.h"

/*
 * Fitness assignment scheme which use a metric
 */
template < class MOEOT>
class moeoMetricFitnessAssignment : public moeoSingleObjectivization < MOEOT >
{
	public:


		/** the objective vector type of the solutions */
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename MOEOT::Fitness Fitness;
		typedef typename ObjectiveVector::Type Type;


		/**
		 * Default ctor
		 * @param _metric a metric function to calculate fitness
		 * @param _negate true if fitness should be maxed
		 */
		moeoMetricFitnessAssignment(moeoUnaryMetric<ObjectiveVector,Fitness> &_metric, bool _negate=true) : metric(_metric),eval(defaultEval),negate(_negate)
	{}
		/**
		 * ctor with an evaluation fonction, applied if give moeot is invalid
		 * @param _eval a evalFunc to regenerate the objectiveVector if needed
		 * @param _metric a metric function to calculate fitness
		 * @param _negate true if fitness should be maxed
		 */
		moeoMetricFitnessAssignment(moeoUnaryMetric<ObjectiveVector,Fitness> &_metric,eoEvalFunc<MOEOT> &_eval, bool _negate=false) : metric(_metric),eval(_eval),negate(_negate)
	{}

		 /** 
		  * Sets the fitness values for a moeot
		  * @param _mo the MOEOT
		  */
		void operator()(MOEOT &  _mo){
			if (_mo.invalidObjectiveVector()) eval(_mo);
			_mo.fitness(operator()(_mo.objectiveVector()));

		}

		/**
		  return the fitness of a valid objectiveVector
		  @param _mo the objectiveVector
		  @return the fitness value of _mo
		  */
		typename MOEOT::Fitness operator()(const typename MOEOT::ObjectiveVector &  _mo){
			if (negate)
			return 1/metric(_mo);
			else return metric(_mo);
		}


		/**
		 * Sets the fitness values for every solution contained in the population _pop  (and in the archive)
		 * @param _pop the population
		 */
		void operator()(eoPop < MOEOT > & _pop)
		{
			unsigned int pop_size= _pop.size();
			for (unsigned int k=0; k<pop_size; k++){
				operator()(_pop[k]);
			}
		}




		/**
		 * Warning: no yet implemented: Updates the fitness values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
		 * @param _pop the population
		 * @param _objVec the objective vector
		 */
		void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
		{
			//std::cout << "WARNING : updateByDeleting not implemented in moeoAssignmentFitnessAssignment" << std::endl;
		}


	private:

		moeoUnaryMetric<typename MOEOT::ObjectiveVector,Fitness> &metric;
		eoEvalFunc<MOEOT> &eval;
		bool negate;
		class DummyEval: public eoEvalFunc<MOEOT>{
			void operator()(MOEOT &moeo){
			}
		} defaultEval;

};

#endif /*MOEOMETRICFITNESSASSIGNMENT_H_*/
