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

#include <eo>

#include <comparator/moeoObjectiveObjectiveVectorComparator.h>
#include <comparator/moeoParetoObjectiveVectorComparator.h>
#include <core/moeoRealObjectiveVector.h>

template < class ObjectiveVectorTraits >
class moeoDualRealObjectiveVector : public moeoScalarObjectiveVector<ObjectiveVectorTraits, eoMinimizingDualFitness >
{
    protected:
        bool _is_feasible;

    public:

        moeoDualRealObjectiveVector(double value=0.0) : moeoScalarObjectiveVector<ObjectiveVectorTraits, eoMinimizingDualFitness >(value, false) {}

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

        bool dominates(const moeoRealObjectiveVector < ObjectiveVectorTraits > & other) const
        {
            // am I better than the other ?

            // if I'm feasible and the other is not
            if( this->is_feasible() && !other.is_feasible() ) {
                // no, the other has a better objective
                return true;

            } else if( !this->is_feasible() && other.is_feasible() ) {
                // yes, a feasible objective is always better than an unfeasible one
                return false;

            } else {
                // the two objective are of the same type
                // lets rely on the comparator
                moeoParetoObjectiveVectorComparator< moeoDualRealObjectiveVector<ObjectiveVectorTraits> > comparator;
                return comparator(other, *this);
            }
        }

        //! Use when maximizing an 
        bool operator<(const moeoDualRealObjectiveVector < ObjectiveVectorTraits > & other) const
        {
            // am I better than the other ?

            // if I'm feasible and the other is not
            if( this->is_feasible() && !other.is_feasible() ) {
                // no, the other has a better objective
                return true;

            } else if( !this->is_feasible() && other.is_feasible() ) {
                // yes, a feasible objective is always better than an unfeasible one
                return false;

            } else {
                moeoObjectiveObjectiveVectorComparator < moeoDualRealObjectiveVector < ObjectiveVectorTraits > > cmp;
                return cmp(*this, other);
            }
        }
};


#endif /*_DUALREALOBJECTIVEVECTOR_H_*/
