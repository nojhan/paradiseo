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
    Pierre Savéant <pierre.saveant@thalesgroup.com>

*/

#ifndef MOEOPARETODUALOBJECTIVEVECTORCOMPARATOR_H_
#define MOEOPARETODUALOBJECTIVEVECTORCOMPARATOR_H_

#include <comparator/moeoParetoObjectiveVectorComparator.h>

/**
 * This functor class allows to compare 2 objective vectors according to Pareto dominance.
 */
template < class ObjectiveVector >
class moeoParetoDualObjectiveVectorComparator : public moeoParetoObjectiveVectorComparator< ObjectiveVector >
  {
  public:

    /**
     * Returns true if ov1 is dominated by ov2
     * @param _ov1 the first objective vector
     * @param _ov2 the second objective vector
     */
    bool operator()(const ObjectiveVector & ov1, const ObjectiveVector & ov2)
    {
        if( ov1.is_feasible() && !ov2.is_feasible() ) {
            return false;
        } else if( !ov1.is_feasible() && ov2.is_feasible() ) {
            return true;
        } else {
            return moeoParetoObjectiveVectorComparator<ObjectiveVector>::operator()(ov1, ov2);
        }
    }

  };

#endif /*MOEOPARETODUALOBJECTIVEVECTORCOMPARATOR_H_*/
