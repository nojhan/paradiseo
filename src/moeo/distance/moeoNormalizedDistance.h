/*
* <moeoNormalizedDistance.h>
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

#ifndef MOEONORMALIZEDDISTANCE_H_
#define MOEONORMALIZEDDISTANCE_H_

#include <vector>
#include "../../eo/utils/eoRealBounds.h"
#include "moeoDistance.h"

/**
 * The base class for double distance computation with normalized objective values (i.e. between 0 and 1).
 */
template < class MOEOT , class Type = double >
class moeoNormalizedDistance : public moeoDistance < MOEOT , Type >
  {
  public:

    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Default ctr
     */
    moeoNormalizedDistance()
    {
      bounds.resize(ObjectiveVector::Traits::nObjectives());
      // initialize bounds in case someone does not want to use them
      for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
          bounds[i] = eoRealInterval(0,1);
        }
    }


    /**
     * Returns a very small value that can be used to avoid extreme cases (where the min bound == the max bound)
     */
    static double tiny()
    {
      return 1e-6;
    }


    /**
     * Sets the lower and the upper bounds for every objective using extremes values for solutions contained in the population _pop
     * @param _pop the population
     */
    virtual void setup(const eoPop < MOEOT > & _pop)
    {
      double min, max;
      for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
          min = _pop[0].objectiveVector()[i];
          max = _pop[0].objectiveVector()[i];
          for (unsigned int j=1; j<_pop.size(); j++)
            {
              min = std::min(min, _pop[j].objectiveVector()[i]);
              max = std::max(max, _pop[j].objectiveVector()[i]);
            }
          // setting of the bounds for the objective i
          setup(min, max, i);
        }
    }


    /**
     * Sets the lower bound (_min) and the upper bound (_max) for the objective _obj
     * @param _min lower bound
     * @param _max upper bound
     * @param _obj the objective index
     */
    virtual void setup(double _min, double _max, unsigned int _obj)
    {
      if (_min == _max)
        {
          _min -= tiny();
          _max += tiny();
        }
      bounds[_obj] = eoRealInterval(_min, _max);
    }


    /**
     * Sets the lower bound and the upper bound for the objective _obj using a eoRealInterval object
     * @param _realInterval the eoRealInterval object
     * @param _obj the objective index
     */
    virtual void setup(eoRealInterval _realInterval, unsigned int _obj)
    {
      bounds[_obj] = _realInterval;
    }


  protected:

    /** the bounds for every objective (bounds[i] = bounds for the objective i) */
    std::vector < eoRealInterval > bounds;

  };

#endif /*MOEONORMALIZEDDISTANCE_H_*/
