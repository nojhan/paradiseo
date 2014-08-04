/*

(c) 2010 Thales group

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; version 2
    of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact: http://eodev.sourceforge.net

Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>

*/


#ifndef MOEOSCALAROBJECTIVEVECTOR_H_
#define MOEOSCALAROBJECTIVEVECTOR_H_

#include <iostream>
#include <math.h>
#include "../comparator/moeoObjectiveObjectiveVectorComparator.h"
#include "../comparator/moeoParetoObjectiveVectorComparator.h"
#include "moeoObjectiveVector.h"

/**
 * This class allows to represent a solution in the objective space (phenotypic representation) by a std::vector of typed values,
 * i.e. that an objective value is represented using a T, and this for any objective.
 */
template < class ObjectiveVectorTraits, class T >
class moeoScalarObjectiveVector : public moeoObjectiveVector < ObjectiveVectorTraits, T >
{
    public:

        using moeoObjectiveVector < ObjectiveVectorTraits, T >::size;
        using moeoObjectiveVector < ObjectiveVectorTraits, T >::operator[];

        /**
         * Ctor
         */
        moeoScalarObjectiveVector(T _value = 0.0) : moeoObjectiveVector < ObjectiveVectorTraits, T > (_value)
        {}


        /**
         * Ctor from a vector of Ts
         * @param _v the std::vector < T >
         */
        moeoScalarObjectiveVector(std::vector < T > & _v) : moeoObjectiveVector < ObjectiveVectorTraits, T > (_v)
        {}


        /**
         * Returns true if the current objective vector dominates _other according to the Pareto dominance relation
         * (but it's better to use a moeoObjectiveVectorComparator object to compare solutions)
         * @param _other the other moeoScalarObjectiveVector object to compare with
         */
        bool dominates(const moeoScalarObjectiveVector < ObjectiveVectorTraits, T > & _other) const
        {
            moeoParetoObjectiveVectorComparator < moeoScalarObjectiveVector<ObjectiveVectorTraits, T> > comparator;
            return comparator(_other, *this);
        }


        /**
         * Returns true if the current objective vector is equal to _other (according to a tolerance value)
         * @param _other the other moeoScalarObjectiveVector object to compare with
         */
        bool operator==(const moeoScalarObjectiveVector < ObjectiveVectorTraits, T > & _other) const
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
         * @param _other the other moeoScalarObjectiveVector object to compare with 
         */
        bool operator!=(const moeoScalarObjectiveVector < ObjectiveVectorTraits, T > & _other) const
        {
            return ! operator==(_other);
        }


        /**
         * Returns true if the current objective vector is smaller than _other on the first objective, then on the second, and so on
         * (can be usefull for sorting/printing)
         * @param _other the other moeoScalarObjectiveVector object to compare with
         */
        bool operator<(const moeoScalarObjectiveVector < ObjectiveVectorTraits, T > & _other) const
        {
            moeoObjectiveObjectiveVectorComparator < moeoScalarObjectiveVector < ObjectiveVectorTraits, T > > cmp;
            return cmp(*this, _other);
        }


        /**
         * Returns true if the current objective vector is greater than _other on the first objective, then on the second, and so on
         * (can be usefull for sorting/printing)
         * @param _other the other moeoScalarObjectiveVector object to compare with
         */
        bool operator>(const moeoScalarObjectiveVector < ObjectiveVectorTraits, T > & _other) const
        {
            return _other < *this;
        }


        /**
         * Returns true if the current objective vector is smaller than or equal to _other on the first objective, then on the second, and so on
         * (can be usefull for sorting/printing)
         * @param _other the other moeoScalarObjectiveVector object to compare with
         */
        bool operator<=(const moeoScalarObjectiveVector < ObjectiveVectorTraits, T > & _other) const
        {
            return operator==(_other) || operator<(_other);
        }


        /**
         * Returns true if the current objective vector is greater than or equal to _other on the first objective, then on the second, and so on
         * (can be usefull for sorting/printing)
         * @param _other the other moeoScalarObjectiveVector object to compare with
         */
        bool operator>=(const moeoScalarObjectiveVector < ObjectiveVectorTraits, T > & _other) const
        {
            return operator==(_other) || operator>(_other);
        }

};


/**
 * Output for a moeoScalarObjectiveVector object
 * @param _os output stream
 * @param _objectiveVector the objective vector to write
 */
template < class ObjectiveVectorTraits, class T >
std::ostream & operator<<(std::ostream & _os, const moeoScalarObjectiveVector < ObjectiveVectorTraits, T > & _objectiveVector)
{
  for (unsigned int i=0; i<_objectiveVector.size()-1; i++)
      _os << _objectiveVector[i] << " ";
  _os << _objectiveVector[_objectiveVector.size()-1];
  return _os;
}

/**
 * Input for a moeoScalarObjectiveVector object
 * @param _is input stream
 * @param _objectiveVector the objective vector to read
 */
template < class ObjectiveVectorTraits, class T >
std::istream & operator>>(std::istream & _is, moeoScalarObjectiveVector < ObjectiveVectorTraits, T > & _objectiveVector)
{
  _objectiveVector = moeoScalarObjectiveVector < ObjectiveVectorTraits, T > ();
  for (unsigned int i=0; i<_objectiveVector.size(); i++)
    {
      _is >> _objectiveVector[i];
    }
  return _is;
}

#endif /*MOEOSCALAROBJECTIVEVECTOR_H_*/
