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

#include <comparator/moeoComparator.h>
#include <comparator/moeoFitnessThenDiversityComparator.h>
#include <diversity/moeoDiversityAssignment.h>
#include <diversity/moeoDummyDiversityAssignment.h>
#include <fitness/moeoFitnessAssignment.h>
#include <replacement/moeoReplacement.h>

/**
 * Elitist replacement strategy that consists in keeping the N best individuals.
 */
template < class MOEOT > class moeoElitistReplacement:public moeoReplacement < MOEOT >
{
public:

    /**
     * Full constructor.
     * @param _fitnessAssignment the fitness assignment strategy
     * @param _diversityAssignment the diversity assignment strategy
     * @param _comparator the comparator (used to compare 2 individuals)
     */
    moeoElitistReplacement (moeoFitnessAssignment < MOEOT > & _fitnessAssignment, moeoDiversityAssignment < MOEOT > & _diversityAssignment, moeoComparator < MOEOT > & _comparator) :
            fitnessAssignment (_fitnessAssignment), diversityAssignment (_diversityAssignment), comparator (_comparator)
    {}


    /**
     * Constructor without comparator. A moeoFitThenDivComparator is used as default.
     * @param _fitnessAssignment the fitness assignment strategy
     * @param _diversityAssignment the diversity assignment strategy
     */
    moeoElitistReplacement (moeoFitnessAssignment < MOEOT > & _fitnessAssignment, moeoDiversityAssignment < MOEOT > & _diversityAssignment) :
            fitnessAssignment (_fitnessAssignment), diversityAssignment (_diversityAssignment), comparator (defaultComparator)
    {}


    /**
     * Constructor without moeoDiversityAssignement. A dummy diversity is used as default.
     * @param _fitnessAssignment the fitness assignment strategy
     * @param _comparator the comparator (used to compare 2 individuals)
     */
    moeoElitistReplacement (moeoFitnessAssignment < MOEOT > & _fitnessAssignment, moeoComparator < MOEOT > & _comparator) :
            fitnessAssignment (_fitnessAssignment), diversityAssignment (defaultDiversity), comparator (_comparator)
    {}


    /**
     * Constructor without moeoDiversityAssignement nor moeoComparator.
     * A moeoFitThenDivComparator and a dummy diversity are used as default.
     * @param _fitnessAssignment the fitness assignment strategy
     */
    moeoElitistReplacement (moeoFitnessAssignment < MOEOT > & _fitnessAssignment) :
            fitnessAssignment (_fitnessAssignment), diversityAssignment (defaultDiversity), comparator (defaultComparator)
    {}


    /**
     * Replaces the first population by adding the individuals of the second one, sorting with a moeoComparator and resizing the whole population obtained.
        * @param _parents the population composed of the parents (the population you want to replace)
        * @param _offspring the offspring population
     */
    void operator () (eoPop < MOEOT > &_parents, eoPop < MOEOT > &_offspring)
    {
        unsigned int sz = _parents.size ();
        // merges offspring and parents into a global population
        _parents.reserve (_parents.size () + _offspring.size ());
        std::copy (_offspring.begin (), _offspring.end (), back_inserter (_parents));
        // evaluates the fitness and the diversity of this global population
        fitnessAssignment (_parents);
        diversityAssignment (_parents);
        // sorts the whole population according to the comparator
        std::sort(_parents.begin(), _parents.end(), comparator);
        // finally, resize this global population
        _parents.resize (sz);
        // and clear the offspring population
        _offspring.clear ();
    }


protected:

    /** the fitness assignment strategy */
    moeoFitnessAssignment < MOEOT > & fitnessAssignment;
    /** the diversity assignment strategy */
    moeoDiversityAssignment < MOEOT > & diversityAssignment;
    /** a dummy diversity assignment can be used as default */
    moeoDummyDiversityAssignment < MOEOT > defaultDiversity;
    /** a fitness then diversity comparator can be used as default */
    moeoFitnessThenDiversityComparator < MOEOT > defaultComparator;
    /** this object is used to compare solutions in order to sort the population */
    class Cmp
    {
    public:
        /**
         * Ctor.
         * @param _comp the comparator
         */
        Cmp(moeoComparator < MOEOT > & _comp) : comp(_comp)
        {}
        /**
         * Returns true if _moeo1 is greater than _moeo2 according to the comparator
         * _moeo1 the first individual
         * _moeo2 the first individual
         */
        bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
        {
            return comp(_moeo2,_moeo1);
        }
    private:
        /** the comparator */
        moeoComparator < MOEOT > & comp;
    } comparator;

};

#endif /*MOEOELITISTREPLACEMENT_H_ */
