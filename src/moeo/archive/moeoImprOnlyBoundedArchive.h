/*
* <moeoImprOnlyBoundedArchive.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Lille-Nord Europe, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jeremie Humeau
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
// moeoIMPRONLYBOUNDEDARCHIVE.h
//-----------------------------------------------------------------------------

#ifndef MOEOImprOnlyBoundedArchive_H_
#define MOEOImprOnlyBoundedArchive_H_

#include "moeoBoundedArchive.h"
#include "moeoArchive.h"
/**
 * This class represents a bounded archive which different parameters to specify.
 */
template < class MOEOT >
class moeoImprOnlyBoundedArchive : public moeoBoundedArchive < MOEOT >
{
public:

    using moeoArchive < MOEOT > :: size;
    using moeoArchive < MOEOT > :: resize;
    using moeoArchive < MOEOT > :: operator[];
    using moeoArchive < MOEOT > :: back;
    using moeoArchive < MOEOT > :: pop_back;
    using moeoArchive < MOEOT > :: push_back;
    using moeoArchive < MOEOT > :: begin;
    using moeoArchive < MOEOT > :: end;
    using moeoArchive < MOEOT > :: replace;
    using moeoBoundedArchive < MOEOT > :: maxSize;


    /**
     * The type of an objective vector for a solution
     */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Ctor where you can choose your own moeoComparator and archive size.
     * @param _comparator the functor used to compare objective vectors
     * @param _maxSize the size of archive (must be smaller or egal to the population size)
     * @param _replace boolean which determine if a solution with the same objectiveVector than another one, can replace it or not
     */
    moeoImprOnlyBoundedArchive(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator, unsigned int _maxSize=100, bool _replace=true) : moeoBoundedArchive < MOEOT >(_comparator, _maxSize, _replace){}

    /**
     * Ctor with moeoParetoObjectiveVectorComparator where you can choose your own archive size.
     * @param _maxSize the size of archive (must be smaller or egal to the population size)
     * @param _replace boolean which determine if a solution with the same objectiveVector than another one, can replace it or not
     */
    moeoImprOnlyBoundedArchive(unsigned int _maxSize=100, bool _replace=true) : moeoBoundedArchive < MOEOT >(_maxSize, _replace){}

    /**
     * Updates the archive with a given individual _moeo
     * @param _moeo the given individual
     * @return true if _moeo is non-dominated (and not if it is added to the archive)
     */
    bool operator()(const MOEOT & _moeo)
    {
    	return update(_moeo);
    }


    /**
     * Updates the archive with a given population _pop
     * @param _pop the given population
     * @return true if a _pop[i] is non-dominated (and not if it is added to the archive)
     */
    bool operator()(const eoPop < MOEOT > & _pop)
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
    
    
private:
	/**
	     * Updates the archive with a given individual _moeo *** NEW ***
	     * @param _moeo the given individual
	     */
	    bool update(const MOEOT & _moeo)
	    {
	        // first step: removing the dominated solutions from the archive
	        for (unsigned int j=0; j<size();)
	        {
	            // if the jth solution contained in the archive is dominated by _moeo
	            if ( this->comparator(operator[](j).objectiveVector(), _moeo.objectiveVector()) )
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
	            if ( this->comparator(_moeo.objectiveVector(), operator[](j).objectiveVector()) )
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
	        	if(size()<maxSize)
	        		push_back(_moeo);
	        	else
	        		dom=!dom;
	        }
	        return !dom;
	    }

};

#endif /*MOEOIMPRONLYBOUNDEDARCHIVE_H_*/
