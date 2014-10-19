/*
* <moeoHyperVolumeDifferenceMetric.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Jeremie Humeau
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

#ifndef MOEOHYPERVOLUMEDIFFERENCEMETRIC_H_
#define MOEOHYPERVOLUMEDIFFERENCEMETRIC_H_

#include "moeoMetric.h"
#include "moeoHyperVolumeMetric.h"

/**
 * The contribution metric evaluates the proportion of non-dominated solutions given by a Pareto set relatively to another Pareto set
 * (Meunier, Talbi, Reininger: 'A multiobjective genetic algorithm for radio network optimization', in Proc. of the 2000 Congress on Evolutionary Computation, IEEE Press, pp. 317-324)
 */
template < class ObjectiveVector >
class moeoHyperVolumeDifferenceMetric : public moeoVectorVsVectorBinaryMetric < ObjectiveVector, double >
  {
	public:

    /**
     * Constructor with a coefficient (rho)
     * @param _normalize allow to normalize data (default true)
     * @param _rho coefficient to determine the reference point.
     */
    moeoHyperVolumeDifferenceMetric(bool _normalize=true, double _rho=1.1): normalize(_normalize), rho(_rho), ref_point(/*NULL*/){
        bounds.resize(ObjectiveVector::Traits::nObjectives());
        // initialize bounds in case someone does not want to use them
        for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
            bounds[i] = eoRealInterval(0,1);
        }
    }

    /**
     * Constructor with a reference point
     * @param _normalize allow to normalize data (default true)
     * @param _ref_point the reference point
     */
    moeoHyperVolumeDifferenceMetric(bool _normalize/*=true*/, ObjectiveVector& _ref_point/*=NULL*/): normalize(_normalize), rho(0.0), ref_point(_ref_point){
        bounds.resize(ObjectiveVector::Traits::nObjectives());
        // initialize bounds in case someone does not want to use them
        for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
            bounds[i] = eoRealInterval(0,1);
        }
    }

    /**
     * calculates and returns the HyperVolume value of a pareto front
     * @param _set1 the vector contains all objective Vector of the first pareto front
     * @param _set2 the vector contains all objective Vector of the second pareto front
     */
    virtual double operator()(const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2)
    {

        double hypervolume_set1;
        double hypervolume_set2;

        if(rho >= 1.0){
            //determine bounds
            setup(_set1, _set2);
            //determine reference point
            for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++){
                if(normalize){
                    if (ObjectiveVector::Traits::minimizing(i))
                        ref_point[i]= rho;
                    else
                        ref_point[i]= 1-rho;
                }
                else{
                    if (ObjectiveVector::Traits::minimizing(i))
                        ref_point[i]= bounds[i].maximum() * rho;
                    else
                        ref_point[i]= bounds[i].maximum() * (1-rho);
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

    /**
     * getter on bounds
     * @return bounds
     */
    std::vector < eoRealInterval > getBounds(){
        return bounds;
    }

    /**
     * method calculate bounds for the normalization
     * @param _set1 the vector contains all objective Vector of the first pareto front
     * @param _set2 the vector contains all objective Vector of the second pareto front
     */
    void setup(const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2)
    {
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

            typename ObjectiveVector::Type min, max;
            unsigned int nbObj=ObjectiveVector::Traits::nObjectives();
            bounds.resize(nbObj);
            for (unsigned int i=0; i<nbObj; i++){
                min = _set1[0][i];
                max = _set1[0][i];
                for (unsigned int j=1; j<_set1.size(); j++){
                    min = std::min(min, _set1[j][i]);
                    max = std::max(max, _set1[j][i]);
                }
                for (unsigned int j=0; j<_set2.size(); j++){
                    min = std::min(min, _set2[j][i]);
                    max = std::max(max, _set2[j][i]);
                }
                if( min == max ) {
                    bounds[i] = eoRealInterval(min-tiny(), max+tiny());
                } else {
                    bounds[i] = eoRealInterval(min, max);
                }
            }
        }
    }

    protected:

    /**
     * Returns a very small value that can be used to avoid extreme cases (where the min bound == the max bound)
     */
    static double tiny()
    {
      return 1e-6;
    }

    protected:

    /*boolean indicates if data must be normalized or not*/
    bool normalize;

    double rho;

    /*vectors contains bounds for normalization*/
    std::vector < eoRealInterval > bounds;

    ObjectiveVector ref_point;

  };

#endif /*MOEOHYPERVOLUMEMETRIC_H_*/
