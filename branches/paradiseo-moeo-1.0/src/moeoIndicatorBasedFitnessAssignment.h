// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoIndicatorBasedFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOINDICATORBASEDFITNESSASSIGNMENT_H_
#define MOEOINDICATORBASEDFITNESSASSIGNMENT_H_

#include <eoPop.h>
#include <moeoConvertPopToObjectiveVectors.h>
#include <moeoFitnessAssignment.h>
#include <metric/moeoNormalizedSolutionVsSolutionBinaryMetric.h>
#include <metric/moeoVectorVsSolutionBinaryMetric.h>

/**
 * 
 */
template < class MOEOT >
class moeoIndicatorBasedFitnessAssignment : public moeoFitnessAssignment < MOEOT >
{
public:

	/** The type of objective vector */
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	
	
	/**
	 * Default ctor
	 * @param ...
	 */
	moeoIndicatorBasedFitnessAssignment(moeoVectorVsSolutionBinaryMetric < ObjectiveVector, double > * _metric) : metric(_metric)
	{}
		
	
	/**
	 * Ctor
	 * @param ...
	 */	 
	moeoIndicatorBasedFitnessAssignment(moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > * _solutionVsSolutionMetric, const double _kappa)// : metric(moeoExponentialVectorVsSolutionBinaryMetric < ObjectiveVector > (_solutionVsSolutionMetric, _kappa))
	{
		 metric = new moeoExponentialVectorVsSolutionBinaryMetric < ObjectiveVector > (_solutionVsSolutionMetric, _kappa);
	}
	
	
	/**
	 * 
	 */
	void operator()(eoPop < MOEOT > & _pop)
	{
		eoPop < MOEOT > tmp_pop;
		moeoConvertPopToObjectiveVectors < MOEOT > convertor;
		for (unsigned i=0; i<_pop.size() ; i++)
		{
			tmp_pop.clear();
			tmp_pop = _pop;
			tmp_pop.erase(tmp_pop.begin() + i);
			_pop[i].fitness((*metric) (convertor(tmp_pop), _pop[i].objectiveVector()));
		}
	}




private:
	moeoVectorVsSolutionBinaryMetric < ObjectiveVector, double > * metric;

};

#endif /*MOEOINDICATORBASEDFITNESSASSIGNMENT_H_*/
