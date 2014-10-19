/*
* <moeoBoundedArchive.h>
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
// moeoBoundedArchive.h
//-----------------------------------------------------------------------------

#ifndef MOEOBOUNDEDARCHIVE_H_
#define MOEOBOUNDEDARCHIVE_H_

#include "moeoArchive.h"

/**
 * This class represents a bounded archive which different parameters to specify.
 */
template < class MOEOT >
class moeoBoundedArchive : public moeoArchive < MOEOT >
{
public:
    /**
     * The type of an objective vector for a solution
     */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Ctor where you can choose your own moeoObjectiveVectorComparator and archive size.
     * @param _comparator the functor used to compare objective vectors
     * @param _maxSize the size of archive (must be smaller or egal to the population size)
     * @param _replace boolean which determine if a solution with the same objectiveVector than another one, can replace it or not
     */
    moeoBoundedArchive(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator, unsigned int _maxSize=100, bool _replace=true) : moeoArchive < MOEOT >(_comparator, _replace), maxSize(_maxSize){}

    /**
     * Ctor with moeoParetoObjectiveVectorComparator where you can choose your own archive size.
     * @param _maxSize the size of archive (must be smaller or egal to the population size)
     * @param _replace boolean which determine if a solution with the same objectiveVector than another one, can replace it or not
     */
    moeoBoundedArchive(unsigned int _maxSize=100, bool _replace=true) : moeoArchive < MOEOT >(_replace), maxSize(_maxSize){}

protected:

    /** archive max size */
    unsigned int maxSize;

};

#endif /*MOEOBOUNDEDARCHIVE_H_*/
