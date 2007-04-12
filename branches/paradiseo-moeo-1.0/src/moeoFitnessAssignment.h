// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOFITNESSASSIGNMENT_H_
#define MOEOFITNESSASSIGNMENT_H_

#include <eoFunctor.h>
#include <eoPop.h>

/**
 * Functor that sets the fitness values of a whole population.
 */
template < class MOEOT >
class moeoFitnessAssignment : public eoUF < eoPop < MOEOT > &, void >
{
public:

	/** The type for objective vector */
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;

	
	/**
	 * Updates the fitness values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
	 * @param _pop the population
	 * @param _objecVec the objective vector
	 */
	virtual void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec) = 0;
	
	
	/**
	 * Updates the fitness values of the whole population _pop by taking the deletion of the individual _moeo into account.
	 * @param _pop the population
	 * @param _moeo the individual
	 */
	void updateByDeleting(eoPop < MOEOT > & _pop, MOEOT & _moeo)
	{
		updateByDeleting(_pop, _moeo.objectiveVector());
	}

};


/**
 * moeoDummyFitnessAssignment is a moeoFitnessAssignment that gives the value '0' as the individual's fitness for a whole population if it is invalid.
 */
template < class MOEOT >
class moeoDummyFitnessAssignment : public moeoFitnessAssignment < MOEOT >
{
public:

	/** The type for objective vector */
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	
	
	/**
	 * Sets the fitness to '0' for every individuals of the population _pop if it is invalid
	 * @param _pop the population
	 */
	 void operator () (eoPop < MOEOT > & _pop)
	 {
	 	for (unsigned idx = 0; idx<_pop.size (); idx++)
	 	{
	 		if (_pop[idx].invalidFitness())
	 		{
	 			// set the diversity to 0
	 			_pop[idx].fitness(0.0);
	 		}
	 	}
	 }
	
	 
	/**
	 * Updates the fitness values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
	 * @param _pop the population
	 * @param _objecVec the objective vector
	 */
	void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
	{
		// nothing to do...  ;-)
	}
	 
};


/**
 * moeoScalarFitnessAssignment is a moeoFitnessAssignment for scalar strategies.
 */
template < class MOEOT >
class moeoScalarFitnessAssignment : public moeoFitnessAssignment < MOEOT >
{};


/**
 * moeoCriterionBasedFitnessAssignment is a moeoFitnessAssignment for criterion-based strategies.
 */
template < class MOEOT >
class moeoCriterionBasedFitnessAssignment : public moeoFitnessAssignment < MOEOT >
{};


/**
 * moeoParetoBasedFitnessAssignment is a moeoFitnessAssignment for Pareto-based strategies.
 */
template < class MOEOT >
class moeoParetoBasedFitnessAssignment : public moeoFitnessAssignment < MOEOT >
{};


#endif /*MOEOFITNESSASSIGNMENT_H_*/
