/*

(c) 2013 Thales group

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

#ifndef MOEODUALHYPERVOLUMEDIFFERENCEMETRIC_H_
#define MOEODUALHYPERVOLUMEDIFFERENCEMETRIC_H_

#include <metric/moeoHyperVolumeDifferenceMetric.h>


template<class ObjectiveVector>
class moeoDualHyperVolumeDifferenceMetric : public moeoHyperVolumeDifferenceMetric<ObjectiveVector>
{
protected:
    using moeoHyperVolumeDifferenceMetric<ObjectiveVector>::rho;
    using moeoHyperVolumeDifferenceMetric<ObjectiveVector>::normalize;
    using moeoHyperVolumeDifferenceMetric<ObjectiveVector>::ref_point;
    using moeoHyperVolumeDifferenceMetric<ObjectiveVector>::bounds;

public:

    typedef typename ObjectiveVector::Type Type;

    moeoDualHyperVolumeDifferenceMetric( bool _normalize=true, double _rho=1.1)
        : moeoHyperVolumeDifferenceMetric<ObjectiveVector>(_normalize, _rho)
    {

    }

    moeoDualHyperVolumeDifferenceMetric( bool _normalize/*=true*/, ObjectiveVector& _ref_point/*=NULL*/ )
        : moeoHyperVolumeDifferenceMetric<ObjectiveVector>( _normalize, _ref_point )
    {

    }

    /**
     * calculates and returns the HyperVolume value of a pareto front
     * @param _set1 the vector contains all objective Vector of the first pareto front
     * @param _set2 the vector contains all objective Vector of the second pareto front
     */
    virtual double operator()(const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2)
    {
#ifndef NDEBUG
        // the two sets must be homogeneous in feasibility
        assert( _set1.size() > 0 );
        for( unsigned int i=1; i<_set1.size(); ++i ) {
            assert( _set1[i].is_feasible() == _set1[0].is_feasible() );
        }
        assert( _set2.size() > 0 );
        for( unsigned int i=1; i<_set2.size(); ++i ) {
            assert( _set2[i].is_feasible() == _set2[0].is_feasible() );
        }
        // and they must have the same feasibility
        assert( _set1[0].is_feasible() == _set2[0].is_feasible() );
#endif
        bool feasible = _set1[0].is_feasible();

        double hypervolume_set1;
        double hypervolume_set2;

        if(rho >= 1.0){
            //determine bounds
            setup(_set1, _set2);
            //determine reference point
            for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++){
                if(normalize){
                    if (ObjectiveVector::Traits::minimizing(i))
                        ref_point[i]= Type(rho, feasible);
                    else
                        ref_point[i]= Type(1-rho, feasible);
                }
                else{
                    if (ObjectiveVector::Traits::minimizing(i))
                        ref_point[i]= Type(bounds[i].maximum() * rho, feasible);
                    else
                        ref_point[i]= Type(bounds[i].maximum() * (1-rho), feasible);
                }
            }
            //if no normalization, reinit bounds to O..1 for
            if(!normalize)
                for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
                    bounds[i] = eoRealInterval(0,1);

        }
        else if(normalize)
            setup(_set1, _set2);

        moeoHyperVolumeMetric <ObjectiveVector> unaryMetric(ref_point, bounds);
        hypervolume_set1 = unaryMetric(_set1);
        hypervolume_set2 = unaryMetric(_set2);

        return hypervolume_set1 - hypervolume_set2;
    }
};

#endif /*MOEODUALHYPERVOLUMEDIFFERENCEMETRIC_H_*/
