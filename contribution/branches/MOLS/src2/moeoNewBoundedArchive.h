/*
* <moeoNewBoundedArchive.h>
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

#ifndef MOEONEWBOUNDEDARCHIVE_H_
#define MOEONEWBOUNDEDARCHIVE_H_

#include <comparator/moeoObjectiveVectorComparator.h>
#include <comparator/moeoParetoObjectiveVectorComparator.h>
#include <archive/moeoArchive.h>
#include <moeoDMLSArchive.h>

template < class MOEOT >
class moeoNewBoundedArchive : public moeoDMLSArchive < MOEOT >
{
public:

    using moeoArchive < MOEOT > :: size;
    using moeoArchive < MOEOT > :: operator[];
    using moeoArchive < MOEOT > :: back;
    using moeoArchive < MOEOT > :: pop_back;
    using moeoDMLSArchive < MOEOT > :: isModified;


    /**
     * The type of an objective vector for a solution
     */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Default ctor.
     * The moeoObjectiveVectorComparator used to compare solutions is based on Pareto dominance
     */
 moeoNewBoundedArchive(unsigned int _maxSize=100) : moeoDMLSArchive < MOEOT >(), maxSize(_maxSize)
    {}


    /**
     * Ctor
     * @param _comparator the moeoObjectiveVectorComparator used to compare solutions
     */
 moeoNewBoundedArchive(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator, unsigned int _maxSize=100) : moeoDMLSArchive < MOEOT >(_comparator), maxSize(_maxSize)
    {}



private:

    /** Max size of archive*/
    unsigned int maxSize;

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
            if ( comparator(operator[](j).objectiveVector(), _moeo.objectiveVector()) )
            {
                operator[](j) = back();
                pop_back();
            }
/////////////////////////////////////////////////////////////////////
//             else if (_moeo.objectiveVector() == operator[](j).objectiveVector())
//             {
//                 operator[](j) = back();
//                 pop_back();
//             }
/////////////////////////////////////////////////////////////////////
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
/////////////////////////////////////////////////////////////////////
            else if ( _moeo.objectiveVector() == operator[](j).objectiveVector() )
            {
                dom = true;
                break;
            }
/////////////////////////////////////////////////////////////////////
        }
        if (!dom)
        {
        	if(size()<maxSize){
        		push_back(_moeo);
        		isModified=true;
        	}
        	else
        		dom=!dom;
        }
        return !dom;
    }

};

#endif /*MOEONEWBOUNDEDARCHIVE_H_ */
