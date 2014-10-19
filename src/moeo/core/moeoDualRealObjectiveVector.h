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

#ifndef _DUALREALOBJECTIVEVECTOR_H_
#define _DUALREALOBJECTIVEVECTOR_H_

#include <iostream>
#include <math.h>

#include "../comparator/moeoObjectiveObjectiveVectorComparator.h"
#include "../comparator/moeoParetoObjectiveVectorComparator.h"
#include "moeoScalarObjectiveVector.h"

template < class ObjectiveVectorTraits, class T = eoMaximizingDualFitness /* can be an eoMinimizingDualFitness */>
class moeoDualRealObjectiveVector : public moeoScalarObjectiveVector<ObjectiveVectorTraits, T >
{
    protected:
        bool _is_feasible;

    public:
        typedef ObjectiveVectorTraits Traits;
        typedef T Base;

        using moeoScalarObjectiveVector < ObjectiveVectorTraits, T >::size;
        using moeoScalarObjectiveVector < ObjectiveVectorTraits, T >::operator[];

        moeoDualRealObjectiveVector(double value=0.0, bool feasible = false)
            : moeoScalarObjectiveVector<ObjectiveVectorTraits,T>
              ( T(value, feasible) ) {}

        bool is_feasible() const
        {
#ifndef NDEBUG
            // if the feasibility is correctly assigned,
            // every scalar's feasibility should be equal to the objective vector
            for( typename moeoDualRealObjectiveVector::const_iterator it = this->begin(), end = this->end(); it != end; ++it ) {
                assert( it->is_feasible() == _is_feasible );
            }
#endif
            return _is_feasible;
        }

        //! One should ensure that feasabilities of scalars are all the same
        void is_feasible( bool value )
        {
#ifndef NDEBUG
            for( typename moeoDualRealObjectiveVector::const_iterator it = this->begin(), end = this->end(); it != end; ++it ) {
                assert( it->is_feasible() == value );
            }
#endif
            _is_feasible = value;
        }

        /**
         * Returns true if the current objective vector dominates _other according to the Pareto dominance relation
         */
        bool dominates(const moeoRealObjectiveVector < ObjectiveVectorTraits > & other) const
        {
            moeoParetoDualObjectiveVectorComparator<moeoDualRealObjectiveVector> cmp;
            return cmp( other, *this );
        }

        //! True if this is smaller than other
        bool operator<(const moeoDualRealObjectiveVector < ObjectiveVectorTraits > & other) const
        {
            // am I better than the other ?

            // if I'm feasible and the other is not
            if( this->is_feasible() && !other.is_feasible() ) {
                // no, the other has a better objective
                return false;

            } else if( !this->is_feasible() && other.is_feasible() ) {
                // yes, a feasible objective is always better than an unfeasible one
                return true;

            } else {
                moeoObjectiveObjectiveVectorComparator < moeoDualRealObjectiveVector < ObjectiveVectorTraits > > cmp;
                // Returns true if this is smaller than other
                return cmp(*this, other);
            }
        }
};



/**
 * Output for a moeoDualRealObjectiveVector object
 * @param _os output stream
 * @param _objectiveVector the objective vector to write
 */
template<class ObjectiveVectorTraits, class T>
std::ostream & operator<<( std::ostream & _os, const moeoDualRealObjectiveVector<ObjectiveVectorTraits,T> & _objectiveVector )
{
    for( unsigned int i=0; i<_objectiveVector.size()-1; i++ ) {
        _os << _objectiveVector[i] << " ";
    }
    _os << _objectiveVector[_objectiveVector.size()-1];
    return _os;
}

/**
 * Input for a moeoDualRealObjectiveVector object
 * @param _is input stream
 * @param _objectiveVector the objective vector to read
 */
template<class ObjectiveVectorTraits, class T>
std::istream & operator>>( std::istream & _is, moeoDualRealObjectiveVector<ObjectiveVectorTraits,T> & _objectiveVector )
{
    _objectiveVector = moeoDualRealObjectiveVector<ObjectiveVectorTraits,T> ();
    for( unsigned int i=0; i<_objectiveVector.size(); i++ ) {
        _is >> _objectiveVector[i];
    }
    return _is;
}


#endif /*_DUALREALOBJECTIVEVECTOR_H_*/
