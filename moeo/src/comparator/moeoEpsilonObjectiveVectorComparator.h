/*
* <moeoEpsilonObjectiveVectorComparator.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
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

#ifndef MOEOEPSILONOBJECTIVEVECTORCOMPARATOR_H_
#define MOEOEPSILONOBJECTIVEVECTORCOMPARATOR_H_

#include <comparator/moeoObjectiveVectorComparator.h>

/**
 * This functor class allows to compare 2 objective vectors according to epsilon dominance.
 */
template < class ObjectiveVector >
class moeoEpsilonObjectiveVectorComparator : public moeoObjectiveVectorComparator < ObjectiveVector >
{
public:

    /**
     * Ctor.
     * @param _epsilon the epsilon value
     */
    moeoEpsilonObjectiveVectorComparator(double _epsilon) : epsilon(_epsilon)
    {}

    /**
     * Returns true if _objectiveVector1 is epsilon-dominated by _objectiveVector2
     * @param _objectiveVector1 the first objective vector
     * @param _objectiveVector2 the second objective vector
     */
    bool operator()(const ObjectiveVector & _objectiveVector1, const ObjectiveVector & _objectiveVector2)
    {
        for (unsigned int i=0; i<ObjectiveVector::nObjectives(); i++)
        {
            //  _objectiveVector1[i] == _objectiveVector2[i] ?
            if ( fabs((_objectiveVector1[i]/ epsilon) - _objectiveVector2[i]) > ObjectiveVector::Traits::tolerance() )
            {
                if (ObjectiveVector::minimizing(i))
                {
                    if ((_objectiveVector1[i] / epsilon) <= _objectiveVector2[i])
                    {
                        return false;		// _objectiveVector1[i] is not better than _objectiveVector2[i]
                    }
                }
                else if (ObjectiveVector::maximizing(i))
                {
                    if ((_objectiveVector1[i] / epsilon) >= _objectiveVector2[i])
                    {
                        return false;		// _objectiveVector1[i] is not better than _objectiveVector2[i]
                    }
                }
            }
        }
        return true;
    }

private:
    /** the reference point */
    double epsilon;
};

#endif /*MOEOEPSILONOBJECTIVEVECTORCOMPARATOR_H_*/
