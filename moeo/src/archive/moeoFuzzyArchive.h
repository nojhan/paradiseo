/*
  <moeoFuzzyArchive.h>
   Oumayma BAHRI

Author:
       Oumayma BAHRI <oumaymabahri.com>

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr	   

*/
//-----------------------------------------------------------------------------
#ifndef MOEOFUZZYARCHIVE_H_
#define MOEOFUZZYARCHIVE_H_

#include <eoPop.h>
#include <comparator/moeoFuzzyParetoComparator.h>


/**
 * Abstract class for representing an archive in a fuzzy space of solutions;
 * an archive is a secondary population that stores non-dominated fuzzy solutions.
 */
template < class MOEOT >
class moeoFuzzyArchive : public eoPop < MOEOT >, public eoUF < const MOEOT &, bool>, public eoUF < const eoPop < MOEOT > &, bool>
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
     * ctor.
     * The FuzzyComparator is used to compare fuzzy solutions based on Pareto dominance
     * @param _replace boolean which determine if a solution with the same objectiveVector than another one, can replace it or not
     */
    moeoFuzzyArchive(FuzzyComparator< ObjectiveVector > & _comparator, bool _replace=true) : eoPop < MOEOT >(), comparator(_comparator), replace(_replace)
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
     * @return if the _moeo is added to the archive
     */
    virtual bool operator()(const MOEOT & _moeo) = 0;


    /**
     * Updates the archive with a given population _pop
     * @param _pop the given population
     * @return if at least one _pop[i] is added to the archive
     */
    virtual bool operator()(const eoPop < MOEOT > & _pop) = 0;


    /**
     * Returns true if the current archive contains the same objective vectors than the given archive _arch
     * @param _arch the given archive
     */
    bool equals (const moeoFuzzyArchive < MOEOT > & _arch)
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

protected:
	/**
     * Updates the archive with a given individual _moeo
     * @param _moeo the given individual
     */
    bool update(const MOEOT & _moeo)
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
            else if (replace && (_moeo.objectiveVector() == operator[](j).objectiveVector()))
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
            else if (!replace && (_moeo.objectiveVector() == operator[](j).objectiveVector()) )
            {
            	dom = true;
            	break;
            }
        }
        return !dom;
    }


    /**
     * Updates the archive with a given population _pop
     * @param _pop the given population
     */
    bool update(const eoPop < MOEOT > & _pop)
    {
    	bool res = false;
    	bool tmp = false;
        for (unsigned int i=0; i<_pop.size(); i++)
        {
            tmp = (*this).update(_pop[i]);
            res = tmp || res;
        }
        return res;
    }

    /** A comparator based on fuzzy Pareto dominance (used as default) */
    moeoFuzzyParetoComparator < ObjectiveVector > FuzzyComparator;
	/** boolean */
	bool replace;
};

#endif /*MOEOFUZZYARCHIVE_H_ */
