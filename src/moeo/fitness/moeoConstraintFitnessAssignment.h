/*
 * <moeoConstraintFitnessAssignment.h>
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
// moeoConstraintFitnessAssignment.h
//-----------------------------------------------------------------------------
#ifndef MOEOCONSTRAINTFITNESSASSIGNMENT_H_
#define MOEOCONSTRAINTFITNESSASSIGNMENT_H_

#include "../../eo/eoPop.h"
#include "moeoSingleObjectivization.h"
#include "../utils/moeoObjectiveVectorNormalizer.h"

/*
 * Fitness assignment scheme which give a penalty if MOEOT does not respect constraints
 */
template < class MOEOT >
class moeoConstraintFitnessAssignment : public moeoSingleObjectivization < MOEOT >
{
	public:

		/** the objective vector type of the solutions */
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		/** the fitness type of the solutions */
		typedef typename MOEOT::Fitness Fitness;
		/** the type of the solutions */
		typedef typename ObjectiveVector::Type Type;

		/**
		 * Default ctor
		 * @param _weight vectors contains all weights to apply for not respecting the contraint in each dimension.
		 * @param _constraint vector containing the constraints, normalizer is applied to it
		 * @param _to_optimize dimension in which we ignore the constraint
		 * @param _normalizer normalizer to apply to each objective
		 */
		moeoConstraintFitnessAssignment(std::vector<double> & _weight, ObjectiveVector &_constraint, int _to_optimize, moeoObjectiveVectorNormalizer<MOEOT> &_normalizer, eoEvalFunc<MOEOT> &_eval) : weight(_weight),constraint(_constraint),to_optimize(_to_optimize),normalizer(_normalizer),eval(_eval),to_eval(true){}

		/**
		 * Ctor with a dummy eval
		 * @param _weight vectors contains all weights to apply for not respecting the contraint in each dimension.
		 * @param _constraint vector containing the constraints, normalizer is applied to it
		 * @param _to_optimize dimension in which we ignore the constraint
		 * @param _normalizer normalizer to apply to each objective
		 */
		moeoConstraintFitnessAssignment(std::vector<double> & _weight, ObjectiveVector &_constraint, int _to_optimize, moeoObjectiveVectorNormalizer<MOEOT> &_normalizer) : weight(_weight), constraint(_constraint), to_optimize(_to_optimize), normalizer(_normalizer), eval(defaultEval), to_eval(false){}

		/** 
		 * Sets the fitness values for every solution contained in the population _pop  (and in the archive)
		 * @param _mo the MOEOT
		 */
		void operator()(MOEOT &  _mo){
			if (to_eval && _mo.invalidObjectiveVector())
				eval(_mo);
			_mo.fitness(operator()(_mo.objectiveVector()));
		}

		/**
		 * Calculate a fitness from a valid objectiveVector
		 * @param _mo a valid objectiveVector
		 * @return the fitness of _mo
		 */
		Fitness operator()(const typename MOEOT::ObjectiveVector &  _mo){
			unsigned int dim=_mo.nObjectives();
			Fitness res=0;
			if (dim>weight.size()){
				std::cout<<"moeoAggregationFitnessAssignmentFitness: ouch, given weight dimension is smaller than MOEOTs"<<std::endl;
			}
			else{
				for(unsigned int l=0; l<dim; l++){
					if ((int)l==to_optimize)
						if (_mo.minimizing(l))
							res-=(normalizer(_mo)[l]) * weight[l];
						else
							res+=(normalizer(_mo)[l]) * weight[l];
					else{
						if(_mo.minimizing(l)){
							if (normalizer(_mo)[l]>normalizer(constraint)[l])
								res-=(normalizer(_mo)[l]-normalizer(constraint)[l])*weight[l];
						}
						else{
							if (normalizer(_mo)[l]<normalizer(constraint)[l])
								//negative so we add it instead of removing it
								res+=(normalizer(_mo)[l]-normalizer(constraint)[l])*weight[l];
						}
					}
				}
			}
			return res;
		}

		/**
		 * Sets the fitness values for every solution contained in the population _pop  (and in the archive)
		 * @param _pop the population
		 */
		void operator()(eoPop < MOEOT > & _pop)
		{
			for(unsigned int k=0; k<_pop.size(); k++)
				operator()(_pop[k]);
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

		//dummy evaluation function
		class DummyEval: public eoEvalFunc<MOEOT>{
			void operator()(MOEOT &moeo){
			}
		} defaultEval;

		//the vector of weight
		std::vector<double> weight;

		//the vector of constraints
		ObjectiveVector constraint;

		//index of the objective to optimize
		int to_optimize;

		//the normalizer
		moeoObjectiveVectorNormalizer<MOEOT>& normalizer;

		//the evaluation function
		eoEvalFunc<MOEOT> &eval;

		//true if the evaluation has to be done
		bool to_eval;

};

#endif /*MOEOAGGREGATIONFITNESSASSIGNMENT_H_*/
