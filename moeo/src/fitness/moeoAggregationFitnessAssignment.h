/*
 * <moeoAggregationFitnessAssignment.h>
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
// moeoAggregationFitnessAssignment.h
//-----------------------------------------------------------------------------
#ifndef MOEOAGGREGATIONFITNESSASSIGNMENT_H_
#define MOEOAGGREGATIONFITNESSASSIGNMENT_H_

#include <eoPop.h>
#include <eoEvalFunc.h>
#include <fitness/moeoSingleObjectivization.h>

/*
 * Fitness assignment scheme which use weight for each objective
 */
template < class MOEOT >
class moeoAggregationFitnessAssignment : public moeoSingleObjectivization < MOEOT >
{
	public:

		/** the objective vector type of the solutions */
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename MOEOT::Fitness Fitness;

		/**
		 * Default ctor
		 * @param _weight vectors contains all weights.
		 * @param _eval a eval function, to revalidate the objectiveVector if needed
		 */
		moeoAggregationFitnessAssignment(std::vector<double> & _weight,eoEvalFunc<MOEOT> &_eval) : weight(_weight),eval(_eval){}

		/**
		 * Ctor with a dummy evaluation function
		 * @param _weight vectors contains all weights.
		 */
		moeoAggregationFitnessAssignment(std::vector<double> & _weight) : weight(_weight),eval(defaultEval){}

		/** 
		 * Sets the fitness values for _moeot
		 * @param _moeot the MOEOT
		 */
		virtual void operator()(MOEOT &  _moeot){
			if (_moeot.invalidObjectiveVector())
				eval(_moeot);
			_moeot.fitness(operator()(_moeot.objectiveVector()));
		}

		/**
		 * function which calculate the fitness from an objectiveVector (which has troi be valid.)
		 * @param _mo an valid objectiveVector
		 * @return the fitness value of _mo
		 */
		virtual Fitness operator()(const typename MOEOT::ObjectiveVector &  _mo){
			unsigned int dim=_mo.nObjectives();
			Fitness res=0;
			if (dim>weight.size()){
				std::cout<<"moeoAggregationFitnessAssignmentFitness: Error -> given weight dimension is smaller than MOEOTs"<<std::endl;
				return res;
			}
			for(unsigned int l=0; l<dim; l++){
				if (_mo.minimizing(l))
					res-=(_mo[l]) * weight[l];
				else
					res+=(_mo[l]) * weight[l];
			}
			return res;
		}

		/**
		 * Sets the fitness values for every solution contained in the population _pop  (and in the archive)
		 * @param _pop the population
		 */
		virtual void operator()(eoPop < MOEOT > & _pop){
			for (unsigned int k=0; k < _pop.size(); k++)
				operator()(_pop[k]);
		}

		/**
		 * Warning: no yet implemented: Updates the fitness values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
		 * @param _pop the population
		 * @param _objVec the objective vector
		 */
		void updateByDeleting(eoPop < MOEOT > & /*_pop*/, ObjectiveVector & /*_objVec*/){}

	private:

		class DummyEval: public eoEvalFunc<MOEOT>{
			void operator()(MOEOT &/*moeo*/){}
		}defaultEval;

		//the vector of weight
		std::vector<double>& weight;
		eoEvalFunc<MOEOT>& eval;

};
#endif /*MOEOAGGREGATIONFITNESSASSIGNMENT_H_*/
