/*
 * <moeoScalarizationFunctionMetricFitnessAssignment.h>
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
// moeoScalarizationFunctionMetricFitnessAssignment.h
//-----------------------------------------------------------------------------
#ifndef MOEOASFAFITNESSASSIGNMENT_H_
#define MOEOASFAFITNESSASSIGNMENT_H_

#include "../../../eo/eoPop.h"
#include "../../fitness/moeoSingleObjectivization.h"
#include "../distance/moeoAchievementScalarizingFunctionDistance.h"
#include "../../metric/moeoDistanceMetric.h"
#include "../../utils/moeoObjectiveVectorNormalizer.h"

/*
 * Fitness assignment scheme which use a metric
 */
template < class MOEOT>
class moeoAchievementScalarizingFunctionMetricFitnessAssignment : public moeoSingleObjectivization < MOEOT >
{
	public:

		/** the objective vector type of the solutions */
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename MOEOT::Fitness Fitness;
		typedef typename ObjectiveVector::Type Type;

		/**
		 * ctor with normalizer
		 * @param _rho
		 * @param _reference the reference point
		 * @param _weight the weights applied to the objectives
		 * @param _normalizer the normalizer to apply to objectiveVectors
		 */
		moeoAchievementScalarizingFunctionMetricFitnessAssignment(unsigned int _rho, const ObjectiveVector& _reference, const ObjectiveVector& _weight, moeoObjectiveVectorNormalizer<MOEOT>& _normalizer) : normalizer(_normalizer), eval(defaultEval), distance(_rho, _weight),	metric( distance, _reference, defaultNormalizer){}

		/**
		 * ctor with an evaluing fonction, applied if give moeot is invalid
		 * @param _rho
		 * @param _reference the reference point
		 * @param _weight the weights applied to the objectives
		 * @param _eval a evalFunc to regenerate the objectiveVector if needed
		 */
		moeoAchievementScalarizingFunctionMetricFitnessAssignment(unsigned int _rho, ObjectiveVector& _reference, const ObjectiveVector& _weight, eoEvalFunc<MOEOT>& _eval): normalizer(defaultNormalizer), eval(_eval), distance(_rho, _weight),	metric( distance, _reference, defaultNormalizer){}

		/**
		 * ctor with an evaluing fonction, applied if give moeot is invalid, and a noramlizer, applied to ObjectiveVectors
		 * @param _rho
		 * @param _reference the reference point
		 * @param _weight the weights applied to the objectives
		 * @param _normalizer the normalizer to apply to objectiveVectors
		 * @param _eval a evalFunc to regenerate the objectiveVector if needed
		 */
		moeoAchievementScalarizingFunctionMetricFitnessAssignment(unsigned int _rho, const ObjectiveVector& _reference, const ObjectiveVector& _weight, moeoObjectiveVectorNormalizer<MOEOT>& _normalizer, eoEvalFunc<MOEOT>& _eval) :	normalizer(_normalizer), eval(_eval), distance(_rho, _weight), metric(distance, _reference, _normalizer){}

		/**
		  default constructor
		 * @param _rho
		 * @param _reference the reference point
		 * @param _weight the weights applied to the objectives
		 */
		moeoAchievementScalarizingFunctionMetricFitnessAssignment(unsigned int _rho, const ObjectiveVector& _reference, const ObjectiveVector& _weight) : normalizer(defaultNormalizer), eval(defaultEval), distance(_rho, _weight), metric(distance, _reference, defaultNormalizer){}

		/** 
		 * Sets the fitness values for a moeot
		 * @param _mo the MOEOT
		 */
		void operator()(MOEOT &  _mo){
			if (_mo.invalidObjectiveVector())
				eval(_mo);
			_mo.fitness(operator()(_mo.objectiveVector()));
		}

		/**
		  return the fitness of a valid objectiveVector
		  @param _mo the objectiveVector
		  @return the fitness value of _mo
		 */
		typename MOEOT::Fitness operator()(const typename MOEOT::ObjectiveVector& _mo){
			return -metric(_mo);
		}

		/**
		 * Sets the fitness values for every solution contained in the populing _pop  (and in the archive)
		 * @param _pop the populing
		 */
		void operator()(eoPop < MOEOT > & _pop)
		{
			for (unsigned int k=0; k < _pop.size(); k++)
				operator()(_pop[k]);
		}

		/**
		 * @param _pop the populing
		 * @param _objVec the objective vector
		 */
		void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec){}

	private:

		class DummyEval: public eoEvalFunc<MOEOT>{
			void operator()(MOEOT &moeo){
			}
		} defaultEval;

		moeoObjectiveVectorNormalizer<MOEOT> &normalizer;
		moeoObjectiveVectorNormalizer<MOEOT> defaultNormalizer;
		eoEvalFunc<MOEOT> &eval;
		moeoAchievementScalarizingFunctionDistance<MOEOT> distance;
		moeoDistanceMetric<MOEOT> metric;


};

#endif /*moeoAugmentedScalarizingFunctionMetricFitnessASSIGNMENT_H_*/
