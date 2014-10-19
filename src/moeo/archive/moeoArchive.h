/*
* <moeoArchive.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
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

#ifndef MOEOARCHIVE_H_
#define MOEOARCHIVE_H_

#include "../../eo/eoPop.h"
#include "../comparator/moeoObjectiveVectorComparator.h"
#include "../comparator/moeoParetoObjectiveVectorComparator.h"


/**
 * Abstract class for representing an archive ;
 * an archive is a secondary population that stores non-dominated solutions.
 */
template < class MOEOT >
class moeoArchive : public eoPop < MOEOT >, public eoUF < const MOEOT &, bool>, public eoUF < const eoPop < MOEOT > &, bool>
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
     * @param _replace boolean which determine if a solution with the same objectiveVector than another one, can replace it or not
     */
    moeoArchive(bool _replace=true) : eoPop < MOEOT >(), comparator(paretoComparator), replace(_replace)
    {}


    /**
     * Ctor
     * @param _comparator the moeoObjectiveVectorComparator used to compare solutions
     * @param _replace boolean which determine if a solution with the same objectiveVector than another one, can replace it or not
     */
    moeoArchive(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator, bool _replace=true) : eoPop < MOEOT >(), comparator(_comparator), replace(_replace)
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
        if (!dom)
        {
            this->push_back(_moeo);
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

    /** The moeoObjectiveVectorComparator used to compare solutions */
    moeoObjectiveVectorComparator < ObjectiveVector > & comparator;
    /** A moeoObjectiveVectorComparator based on Pareto dominance (used as default) */
    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;
	/** boolean */
	bool replace;
};

#endif /*MOEOARCHIVE_H_ */
