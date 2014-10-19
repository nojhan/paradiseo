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

#include "moeoHyperVolumeDifferenceMetric.h"


template<class ObjectiveVector>
class moeoDualHyperVolumeDifferenceMetric : public moeoHyperVolumeDifferenceMetric<ObjectiveVector>
{
protected:
    using moeoHyperVolumeDifferenceMetric<ObjectiveVector>::rho;
    using moeoHyperVolumeDifferenceMetric<ObjectiveVector>::tiny;
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
     * method calculate bounds for the normalization
     * @param _set1 the vector contains all objective Vector of the first pareto front
     * @param _set2 the vector contains all objective Vector of the second pareto front
     */
    void setup(const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2)
    {
        typename ObjectiveVector::Type::Compare cmp;

        if(_set1.size() < 1 || _set2.size() < 1) {
            throw("Error in moeoHyperVolumeUnaryMetric::setup -> argument1: vector<ObjectiveVector> size must be greater than 0");
        } else {
#ifndef NDEBUG
            if( _set1.size() == 1 || _set2.size() == 1 ) {
                eo::log << eo::warnings << "Warning in moeoHyperVolumeUnaryMetric::setup one of the pareto set contains only one point (set1.size="
                    << _set1.size() << ", set2.size=" << _set2.size() << ")"
                    << std::endl;
            }
#endif

            typename ObjectiveVector::Type  worst,  best;
            unsigned int nbObj=ObjectiveVector::Traits::nObjectives();
            bounds.resize(nbObj);
            for (unsigned int i=0; i<nbObj; i++){
                worst = _set1[0][i];
                best = _set1[0][i];
                for (unsigned int j=1; j<_set1.size(); j++){
                    worst = std::min( worst, _set1[j][i], cmp );
                    best = std::max( best, _set1[j][i], cmp );
                }
                for (unsigned int j=0; j<_set2.size(); j++){
                    worst = std::min( worst, _set2[j][i], cmp );
                    best = std::max( best, _set2[j][i], cmp );
                }

                // Get real min/max
                double min = std::min(worst.value(), best.value());
                double max = std::max(worst.value(), best.value());

                // Build a fitness with them
                assert( best.is_feasible() == worst.is_feasible() ); // we are supposed to work on homogeneous pop
                Type fmin( min, best.is_feasible() );
                Type fmax( max, best.is_feasible() );

                if(  fmin ==  fmax ) {
                    bounds[i] = eoRealInterval( fmin-tiny(),  fmax+tiny() );
                } else {
                    bounds[i] = eoRealInterval( fmin, fmax );
                }
            } // for i in nbObj
        } // if sizes >= 1
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
