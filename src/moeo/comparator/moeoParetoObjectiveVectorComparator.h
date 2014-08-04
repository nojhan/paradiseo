/*
* <moeoParetoObjectiveVectorComparator.h>
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

#ifndef MOEOPARETOOBJECTIVEVECTORCOMPARATOR_H_
#define MOEOPARETOOBJECTIVEVECTORCOMPARATOR_H_

#include "moeoObjectiveVectorComparator.h"

/**
 * This functor class allows to compare 2 objective vectors according to Pareto dominance.
 */
template < class ObjectiveVector >
class moeoParetoObjectiveVectorComparator : public moeoObjectiveVectorComparator < ObjectiveVector >
  {
  public:

    /**
     * Returns true if _objectiveVector1 is dominated by _objectiveVector2
     * @param _objectiveVector1 the first objective vector
     * @param _objectiveVector2 the second objective vector	
     */
    bool operator()(const ObjectiveVector & _objectiveVector1, const ObjectiveVector & _objectiveVector2)
    {
      bool dom = false;
      for (unsigned int i=0; i<ObjectiveVector::nObjectives(); i++)
        {
          // first, we have to check if the 2 objective values are not equal for the ith objective
          if ( fabs(_objectiveVector1[i] - _objectiveVector2[i]) > ObjectiveVector::Traits::tolerance() )
            {
              // if the ith objective have to be minimized...
              if (ObjectiveVector::minimizing(i))
                {
                  if (_objectiveVector1[i] > _objectiveVector2[i])
                    {
                      dom = true;		//_objectiveVector1[i] is not better than _objectiveVector2[i]
                    }
                  else
                    {
                      return false;	//_objectiveVector2 cannot dominate _objectiveVector1
                    }
                }
              // if the ith objective have to be maximized...
              else if (ObjectiveVector::maximizing(i))
                {
                  if (_objectiveVector1[i] < _objectiveVector2[i])
                    {
                      dom = true;		//_objectiveVector1[i] is not better than _objectiveVector2[i]
                    }
                  else
                    {
                      return false;	//_objectiveVector2 cannot dominate _objectiveVector1
                    }
                }
            }
        }
      return dom;
    }

  };

#endif /*MOEOPARETOOBJECTIVEVECTORCOMPARATOR_H_*/
