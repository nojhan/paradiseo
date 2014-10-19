/*
* <moeoRealObjectiveVector.h>
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

#ifndef MOEOREALOBJECTIVEVECTOR_H_
#define MOEOREALOBJECTIVEVECTOR_H_

#include <iostream>
#include <math.h>
#include "../comparator/moeoObjectiveObjectiveVectorComparator.h"
#include "../comparator/moeoParetoObjectiveVectorComparator.h"
#include "moeoObjectiveVector.h"

/**
 * This class allows to represent a solution in the objective space (phenotypic representation) by a std::vector of real values,
 * i.e. that an objective value is represented using a double, and this for any objective.
 */
template < class ObjectiveVectorTraits >
class moeoRealObjectiveVector : public moeoObjectiveVector < ObjectiveVectorTraits, double >
  {
  public:

    using moeoObjectiveVector < ObjectiveVectorTraits, double >::size;
    using moeoObjectiveVector < ObjectiveVectorTraits, double >::operator[];

    /**
     * Ctor
     */
    moeoRealObjectiveVector(double _value = 0.0) : moeoObjectiveVector < ObjectiveVectorTraits, double > (_value)
    {}


    /**
     * Ctor from a vector of doubles
     * @param _v the std::vector < double >
     */
    moeoRealObjectiveVector(std::vector < double > & _v) : moeoObjectiveVector < ObjectiveVectorTraits, double > (_v)
    {}


    /**
     * Returns true if the current objective vector dominates _other according to the Pareto dominance relation
     * (but it's better to use a moeoObjectiveVectorComparator object to compare solutions)
     * @param _other the other moeoRealObjectiveVector object to compare with
     */
    bool dominates(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
      {
        moeoParetoObjectiveVectorComparator < moeoRealObjectiveVector<ObjectiveVectorTraits> > comparator;
        return comparator(_other, *this);
      }


    /**
     * Returns true if the current objective vector is equal to _other (according to a tolerance value)
     * @param _other the other moeoRealObjectiveVector object to compare with
     */
    bool operator==(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
      {
        for (unsigned int i=0; i < size(); i++)
          {
            if ( fabs(operator[](i) - _other[i]) > ObjectiveVectorTraits::tolerance() )
              {
                return false;
              }
          }
        return true;
      }


    /**
     * Returns true if the current objective vector is different than _other (according to a tolerance value)
     * @param _other the other moeoRealObjectiveVector object to compare with 
     */
    bool operator!=(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
      {
        return ! operator==(_other);
      }


    /**
     * Returns true if the current objective vector is smaller than _other on the first objective, then on the second, and so on
     * (can be usefull for sorting/printing)
     * @param _other the other moeoRealObjectiveVector object to compare with
     */
    bool operator<(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
      {
        moeoObjectiveObjectiveVectorComparator < moeoRealObjectiveVector < ObjectiveVectorTraits > > cmp;
        return cmp(*this, _other);
      }


    /**
     * Returns true if the current objective vector is greater than _other on the first objective, then on the second, and so on
     * (can be usefull for sorting/printing)
     * @param _other the other moeoRealObjectiveVector object to compare with
     */
    bool operator>(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
      {
        return _other < *this;
      }


    /**
     * Returns true if the current objective vector is smaller than or equal to _other on the first objective, then on the second, and so on
     * (can be usefull for sorting/printing)
     * @param _other the other moeoRealObjectiveVector object to compare with
     */
    bool operator<=(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
      {
        return operator==(_other) || operator<(_other);
      }


    /**
     * Returns true if the current objective vector is greater than or equal to _other on the first objective, then on the second, and so on
     * (can be usefull for sorting/printing)
     * @param _other the other moeoRealObjectiveVector object to compare with
     */
    bool operator>=(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
      {
        return operator==(_other) || operator>(_other);
      }

  };


/**
 * Output for a moeoRealObjectiveVector object
 * @param _os output stream
 * @param _objectiveVector the objective vector to write
 */
template < class ObjectiveVectorTraits >
std::ostream & operator<<(std::ostream & _os, const moeoRealObjectiveVector < ObjectiveVectorTraits > & _objectiveVector)
{
  for (unsigned int i=0; i<_objectiveVector.size()-1; i++)
      _os << _objectiveVector[i] << " ";
  _os << _objectiveVector[_objectiveVector.size()-1];
  return _os;
}

/**
 * Input for a moeoRealObjectiveVector object
 * @param _is input stream
 * @param _objectiveVector the objective vector to read
 */
template < class ObjectiveVectorTraits >
std::istream & operator>>(std::istream & _is, moeoRealObjectiveVector < ObjectiveVectorTraits > & _objectiveVector)
{
  _objectiveVector = moeoRealObjectiveVector < ObjectiveVectorTraits > ();
  for (unsigned int i=0; i<_objectiveVector.size(); i++)
    {
      _is >> _objectiveVector[i];
    }
  return _is;
}

#endif /*MOEOREALOBJECTIVEVECTOR_H_*/
