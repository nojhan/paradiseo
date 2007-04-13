// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoEnvironmentalReplacement.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOENVIRONMENTALREPLACEMENT_H_
#define MOEOENVIRONMENTALREPLACEMENT_H_

#include <moeoReplacement.h>
#include <moeoComparator.h>
#include <moeoFitnessAssignment.h>
#include <moeoDiversityAssignment.h>

/**
 * Environmental replacement strategy that consists in keeping the N best individuals by deleting individuals 1 by 1 
 * and by updating the fitness and diversity values after each deletion.
 */
template < class MOEOT > class moeoEnvironmentalReplacement:public moeoReplacement < MOEOT >
{
public:

	/** The type for objective vector */
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	
	
	/**
	 * Full constructor.
	 * @param _evalFitness the fitness assignment strategy
	 * @param _evalDiversity the diversity assignment strategy
	 * @param _comparator the comparator (used to compare 2 individuals)
	 */
	moeoEnvironmentalReplacement (moeoFitnessAssignment < MOEOT > & _evalFitness, moeoDiversityAssignment < MOEOT > & _evalDiversity, moeoComparator < MOEOT > & _comparator) : 
	evalFitness (_evalFitness), evalDiversity (_evalDiversity), comparator (_comparator)
	{}


	/**
	 * Constructor without comparator. A moeoFitThenDivComparator is used as default.
	 * @param _evalFitness the fitness assignment strategy
	 * @param _evalDiversity the diversity assignment strategy
	 */
	moeoEnvironmentalReplacement (moeoFitnessAssignment < MOEOT > & _evalFitness, moeoDiversityAssignment < MOEOT > & _evalDiversity) : 
	evalFitness (_evalFitness), evalDiversity (_evalDiversity), comparator (*(new moeoFitnessThenDiversityComparator < MOEOT >))
	{}


	/**
	 * Constructor without moeoDiversityAssignement. A dummy diversity is used as default.
	 * @param _evalFitness the fitness assignment strategy
	 * @param _comparator the comparator (used to compare 2 individuals)
	 */
	moeoEnvironmentalReplacement (moeoFitnessAssignment < MOEOT > & _evalFitness, moeoComparator < MOEOT > & _comparator) : 
	evalFitness (_evalFitness), evalDiversity (*(new moeoDummyDiversityAssignment < MOEOT >)), comparator (_comparator)
	{}


	/**
	 * Constructor without moeoDiversityAssignement nor moeoComparator.
	 * A moeoFitThenDivComparator and a dummy diversity are used as default.
	 * @param _evalFitness the fitness assignment strategy
	 */
	moeoEnvironmentalReplacement (moeoFitnessAssignment < MOEOT > & _evalFitness) : 
	evalFitness (_evalFitness), evalDiversity (*(new moeoDummyDiversityAssignment < MOEOT >)), comparator (*(new moeoFitnessThenDiversityComparator < MOEOT >))
	{}


	/**
	 * Replaces the first population by adding the individuals of the second one, sorting with a moeoComparator and resizing the whole population obtained.
     * @param _parents the population composed of the parents (the population you want to replace)
     * @param _offspring the offspring population
	 */
	void operator () (eoPop < MOEOT > &_parents, eoPop < MOEOT > &_offspring)
	{
		unsigned sz = _parents.size();
		// merges offspring and parents into a global population
		_parents.reserve (_parents.size() + _offspring.size());
		copy (_offspring.begin(), _offspring.end(), back_inserter(_parents));
		// evaluates the fitness and the diversity of this global population
		evalFitness (_parents);
		evalDiversity (_parents);
		// remove individuals 1 by 1 and update the fitness values
		Cmp cmp(comparator);
		ObjectiveVector worstObjVec;
		while (_parents.size() > sz)
		{
			std::sort (_parents.begin(), _parents.end(), cmp);
			worstObjVec = _parents[_parents.size()-1].objectiveVector();
			_parents.resize(_parents.size()-1);
			evalFitness.updateByDeleting(_parents, worstObjVec);
			evalDiversity.updateByDeleting(_parents, worstObjVec);
		}
		// clear the offspring population
		_offspring.clear ();
	}


protected:

	/** the fitness assignment strategy */
	moeoFitnessAssignment < MOEOT > & evalFitness;
	/** the diversity assignment strategy */
	moeoDiversityAssignment < MOEOT > & evalDiversity;
	/** the comparator (used to compare 2 individuals) */
	moeoComparator < MOEOT > & comparator;


	/**
	 * This class is used to compare solutions in order to sort the population.
	 */
	class Cmp
	{
	public:
	
		/**
		 * Ctor.
		 * @param _comparator the comparator
		 */
		Cmp(moeoComparator < MOEOT > & _comparator) : comparator(_comparator)
		{}
		
		
		/**
		 * Returns true if _moeo1 is greater than _moeo2 according to the comparator
		 * _moeo1 the first individual
		 * _moeo2 the first individual
		 */
		bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
		{
			return comparator(_moeo1,_moeo2);
		}
		
		
	private:
	
		/** the comparator */
		moeoComparator < MOEOT > & comparator;
		
	};

};

#endif /*MOEOENVIRONMENTALREPLACEMENT_H_ */
