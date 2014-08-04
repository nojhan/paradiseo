/*
* <moeoManhattanDistance.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
* Francçois Legillon
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

#ifndef MOEOMANHATTANDISTANCE_H_
#define MOEOMANHATTANDISTANCE_H_

#include <math.h>
#include "moeoObjSpaceDistance.h"
#include "../utils/moeoObjectiveVectorNormalizer.h"

/**
 * A class allowing to compute the Manhattan distance between two solutions in the objective space normalized objective values (i.e. between 0 and 1).
 * A distance value then lies between 0 and nObjectives.
 */
template < class MOEOT >
class moeoManhattanDistance : public moeoObjSpaceDistance < MOEOT >
  {
  public:

    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;
    /** the fitness type of the solutions */
    typedef typename MOEOT::Fitness Fitness;

    /**
      ctr with a normalizer
      @param _normalizer the normalizer used for every ObjectiveVector
      */
    moeoManhattanDistance (moeoObjectiveVectorNormalizer<MOEOT> &_normalizer):normalizer(_normalizer)
    {}
    /**
      default ctr
      */
    moeoManhattanDistance ():normalizer(defaultNormalizer)
    {}

    /**
     * Returns the Manhattan distance between _obj1 and _obj2 in the objective space
     * @param _obj1 the first objective vector
     * @param _obj2 the second objective vector
     */
    double operator()(const ObjectiveVector & _obj1, const ObjectiveVector & _obj2)
    {
      double result = 0.0;
      double tmp1, tmp2;
      for (unsigned int i=0; i<ObjectiveVector::nObjectives(); i++)
        {
          tmp1 = normalizer(_obj1)[i];
          tmp2 = normalizer(_obj2)[i];
          result += fabs(tmp1-tmp2);
        }
      return result;
    }


  private:

    moeoObjectiveVectorNormalizer<MOEOT> defaultNormalizer;
    moeoObjectiveVectorNormalizer<MOEOT> &normalizer;

  };

#endif /*MOEOMANHATTANDISTANCE_H_*/
