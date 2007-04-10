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
 * ???
 */
template < class MOEOT > class moeoEnvironmentalReplacement:public moeoReplacement < MOEOT >
{
public:

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
	evalFitness (_evalFitness), evalDiversity (_evalDiversity)
	{
	    // a moeoFitThenDivComparator is used as default
	    moeoFitnessThenDiversityComparator < MOEOT > &fitThenDivComparator;
	    comparator = fitThenDivComparator;
	}


	/**
	 * Constructor without moeoDiversityAssignement. A dummy diversity is used as default.
	 * @param _evalFitness the fitness assignment strategy
	 * @param _comparator the comparator (used to compare 2 individuals)
	 */
	moeoEnvironmentalReplacement (moeoFitnessAssignment < MOEOT > & _evalFitness, moeoComparator < MOEOT > & _comparator) : 
	evalFitness (_evalFitness), comparator (_comparator)
	{
		// a dummy diversity is used as default
    	moeoDummyDiversityAssignment < MOEOT > &dummyDiversityAssignment;
    	evalDiversity = dummyDiversityAssignment;
	}
	
	/**
	 * Constructor without moeoDiversityAssignement nor moeoComparator.
	 * A moeoFitThenDivComparator and a dummy diversity are used as default.
	 * @param _evalFitness the fitness assignment strategy
	 */
	moeoEnvironmentalReplacement (moeoFitnessAssignment < MOEOT > & _evalFitness) : evalFitness (_evalFitness)
	{
	 	// a dummy diversity is used as default
    	moeoDummyDiversityAssignment < MOEOT > & dummyDiversityAssignment;
    	evalDiversity = dummyDiversityAssignment;
    	// a moeoFitThenDivComparator is used as default
    	moeoFitnessThenDiversityComparator < MOEOT > & fitThenDivComparator;
    	comparator = fitThenDivComparator;
	}

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
		MOEOT worst;
		while (_parents.size() > sz)
		{
			std::sort (_parents.begin(), _parents.end(), cmp);
			worst = _parents[_parents.size()-1];
			_parents.resize(_parents.size()-1);
			evalFitness.updateByDeleting(_parents, worst);
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

	class Cmp
	{
	public:
		Cmp(moeoComparator < MOEOT > & _comparator) : comparator(_comparator) {}
		bool operator()(const MOEOT & a, const MOEOT & b)
		{
			return comparator(a,b);
		}
	private:
		moeoComparator < MOEOT > & comparator;
	};

};

#endif /*MOEOENVIRONMENTALREPLACEMENT_H_ */
