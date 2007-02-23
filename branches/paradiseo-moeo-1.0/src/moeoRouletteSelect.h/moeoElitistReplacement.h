// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoElitistReplacement.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------


#ifndef MOEOELITISTREPLACEMENT_H_
#define MOEOELITISTREPLACEMENT_H_

//-----------------------------------------------------------------------------
#include <moeoReplacement.h>
#include <moeoComparator.h>
#include <moeoFitnessAssignment.h>
#include <moeoDiversityAssignment.h>
//----------------------------------------------------------------------------- 


/**
 * Elitist replacement strategy for multi-objective optimization.
 */
template < class MOEOT > class moeoElitistReplacement:public moeoReplacement <
  MOEOT >
{

public:

	/** Full constructor */
moeoElitistReplacement (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoDiversityAssignment < MOEOT > &_evalDiversity, moeoComparator < MOEOT > _comparator):evalFitness (_evalFitness), evalDiversity (_evalDiversity),
    comparator
    (_comparator)
  {
  }
  
  	/** constructor without comparator */
	moeoElitistReplacement (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoDiversityAssignment < MOEOT > &_evalDiversity)
		:evalFitness (_evalFitness), evalDiversity (_evalDiversity)
	    
	  {
	  	 // a moeoFitThenDivComparator is used as default
	    moeoFitnessThenDiversityComparator < MOEOT > &fitThenDivComparator;
	    comparator = fitThenDivComparator;
	  }


     /** Constructor without moeoDiversityAssignement  */
moeoElitistReplacement (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoComparator < MOEOT > _comparator):evalFitness (_evalFitness),
    comparator
    (_comparator)
  {
    // a dummy diversity is used as default
    moeoDummyDiversityAssignment < MOEOT > &dummyDiversityAssignment;
    evalDiversity = dummyDiversityAssignment;
  }


     /** Constructor without moeoDiversityAssignement nor moeoComparator */
moeoElitistReplacement (moeoFitnessAssignment < MOEOT > &_evalFitness):evalFitness
    (_evalFitness)
  {
    // a dummy diversity is used as default
    moeoDummyDiversityAssignment < MOEOT > &dummyDiversityAssignment;
    evalDiversity = dummyDiversityAssignment;

    // a moeoFitThenDivComparator is used as default
    moeoFitnessThenDiversityComparator < MOEOT > &fitThenDivComparator;
    comparator = fitThenDivComparator;
  }


     /**
      * Replace the first population by adding the individuals of the second one, sorting with a moeoComparator and resizing the whole population obtained.
      * @param _parents the population composed of the parents (the population you want to replace)
      * @param _offspring
      */
  void operator () (eoPop < MOEOT > &_parents, eoPop < MOEOT > &_offspring)
  {
    unsigned sz = _parents.size ();

    // merge offspring and parents into a global population ...
    _parents.reserve (_parents.size () + _offspring.size ());
    copy (_offspring.begin (), _offspring.end (), back_inserter (_parents));

    // evaluate the fitness and the diversity of this global population
    evalFitness (_parents);
    evalDiversity (_parents);

    // ... that we sort according to the comparator
    std::sort (_parents.begin (), _parents.end (), comparator);

    // finally, resize this global population
    _parents.resize (sz);

    _offspring.clear ();
  }

protected:

  moeoFitnessAssignment < MOEOT > &evalFitness;

  moeoDiversityAssignment < MOEOT > &evalDiversity;

  moeoComparator < MOEOT > &comparator;
};



#endif /*MOEOELITISTREPLACEMENT_H_ */
