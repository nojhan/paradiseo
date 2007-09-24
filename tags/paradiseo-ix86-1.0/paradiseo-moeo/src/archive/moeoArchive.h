// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoArchive.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOARCHIVE_H_
#define MOEOARCHIVE_H_

#include <eoPop.h>
#include <comparator/moeoObjectiveVectorComparator.h>
#include <comparator/moeoParetoObjectiveVectorComparator.h>

/**
 * An archive is a secondary population that stores non-dominated solutions.
 */
template < class MOEOT >
class moeoArchive : public eoPop < MOEOT >
{
public:

    using eoPop < MOEOT > :: size;
    using eoPop < MOEOT > :: operator[];
    using eoPop < MOEOT > :: back;
    using eoPop < MOEOT > :: pop_back;


    /**
     * The type of an objective vector for a solution
     */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Default ctor.
     * The moeoObjectiveVectorComparator used to compare solutions is based on Pareto dominance
     */
    moeoArchive() : eoPop < MOEOT >(), comparator(paretoComparator)
    {}


    /**
     * Ctor
     * @param _comparator the moeoObjectiveVectorComparator used to compare solutions
     */
    moeoArchive(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator) : eoPop < MOEOT >(), comparator(_comparator)
    {}


    /**
     * Returns true if the current archive dominates _objectiveVector according to the moeoObjectiveVectorComparator given in the constructor
     * @param _objectiveVector the objective vector to compare with the current archive
     */
    bool dominates (const ObjectiveVector & _objectiveVector) const
    {
        for (unsigned int i = 0; i<size(); i++)
        {
            // if _objectiveVector is dominated by the ith individual of the archive...
            if ( comparator(_objectiveVector, operator[](i).objectiveVector()) )
            {
                return true;
            }
        }
        return false;
    }


    /**
     * Returns true if the current archive already contains a solution with the same objective values than _objectiveVector
     * @param _objectiveVector the objective vector to compare with the current archive
     */
    bool contains (const ObjectiveVector & _objectiveVector) const
    {
        for (unsigned int i = 0; i<size(); i++)
        {
            if (operator[](i).objectiveVector() == _objectiveVector)
            {
                return true;
            }
        }
        return false;
    }


    /**
     * Updates the archive with a given individual _moeo
     * @param _moeo the given individual
     */
    void update (const MOEOT & _moeo)
    {
        // first step: removing the dominated solutions from the archive
        for (unsigned int j=0; j<size();)
        {
            // if the jth solution contained in the archive is dominated by _moeo
            if ( comparator(operator[](j).objectiveVector(), _moeo.objectiveVector()) )
            {
                operator[](j) = back();
                pop_back();
            }
            else if (_moeo.objectiveVector() == operator[](j).objectiveVector())
            {
                operator[](j) = back();
                pop_back();
            }
            else
            {
                j++;
            }
        }
        // second step: is _moeo dominated?
        bool dom = false;
        for (unsigned int j=0; j<size(); j++)
        {
            // if _moeo is dominated by the jth solution contained in the archive
            if ( comparator(_moeo.objectiveVector(), operator[](j).objectiveVector()) )
            {
                dom = true;
                break;
            }
        }
        if (!dom)
        {
            push_back(_moeo);
        }
    }


    /**
     * Updates the archive with a given population _pop
     * @param _pop the given population
     */
    void update (const eoPop < MOEOT > & _pop)
    {
        for (unsigned int i=0; i<_pop.size(); i++)
        {
            update(_pop[i]);
        }
    }


    /**
     * Returns true if the current archive contains the same objective vectors than the given archive _arch
     * @param _arch the given archive
     */
    bool equals (const moeoArchive < MOEOT > & _arch)
    {
        for (unsigned int i=0; i<size(); i++)
        {
            if (! _arch.contains(operator[](i).objectiveVector()))
            {
                return false;
            }
        }
        for (unsigned int i=0; i<_arch.size() ; i++)
        {
            if (! contains(_arch[i].objectiveVector()))
            {
                return false;
            }
        }
        return true;
    }


private:

    /** The moeoObjectiveVectorComparator used to compare solutions */
    moeoObjectiveVectorComparator < ObjectiveVector > & comparator;
    /** A moeoObjectiveVectorComparator based on Pareto dominance (used as default) */
    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;

};

#endif /*MOEOARCHIVE_H_ */
