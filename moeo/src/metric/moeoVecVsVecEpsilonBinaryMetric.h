/*
* <moeoVecVsVecEpsilonBinaryMetric.h>
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

#ifndef MOEOVECVSVECEPSILONBINARYMETRIC_H_
#define MOEOVECVSVECEPSILONBINARYMETRIC_H_

#include <comparator/moeoParetoObjectiveVectorComparator.h>
#include <metric/moeoMetric.h>

/**
 * moeoVecVsVecEpsilonBinaryMetric is an abstract class allow to calculate the epsilon indicator betweend two Pareto sets
 */
template < class ObjectiveVector >
class moeoVecVsVecEpsilonBinaryMetric : public moeoVectorVsVectorBinaryMetric < ObjectiveVector, double >
{
public:

    /**
     * Default Construtcor
     * @param _normalize allow to normalize data (default true)
     */
    moeoVecVsVecEpsilonBinaryMetric(bool _normalize=true): normalize(_normalize){
        bounds.resize(ObjectiveVector::Traits::nObjectives());
        // initialize bounds in case someone does not want to use them
        for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
            bounds[i] = eoRealInterval(0,1);
        }
    }


    /**
     * Returns the epsilon indicator between two pareto sets.
     * @param _set1 the first Pareto set (must not have dominated element)
     * @param _set2 the second Pareto set (must not have dominated element)
     */
    double operator()(const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2)
    {
        if (normalize)
            setup(_set1, _set2);
        double eps, eps_temp, eps_j, eps_k;

        unsigned i, j, k;

        for (i = 0; i < _set2.size(); i++) {
            for (j = 0; j < _set1.size(); j++) {
                for (k = 0; k < ObjectiveVector::Traits::nObjectives(); k++) {
                    eps_temp=epsilon(_set1[j], _set2[i], k);
                    if (k == 0)
                        eps_k = eps_temp;
                    else if (eps_k < eps_temp)
                        eps_k = eps_temp;
                }
                if (j == 0)
                    eps_j = eps_k;
                else if (eps_j > eps_k)
                    eps_j = eps_k;
            }
            if (i == 0)
                eps = eps_j;
            else if (eps < eps_j)
                eps = eps_j;
        }
        return eps;
    }

    std::vector < eoRealInterval > getBounds(){
        return bounds;
    }

    /**
     * method caclulate bounds for the normalization
     * @param _set1 the first vector of objective vectors
     * @param _set2 the second vector of objective vectors
     */
    void setup(const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2){
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
            bounds[i] = eoRealInterval(min, max);
        }
    }


protected:

    /*vectors contains bounds for normalization*/
    std::vector < eoRealInterval > bounds;


    /*boolean indicates if data must be normalized or not*/
    bool normalize;

private :

    /**
     * abstract method allow to use differents epsilon indicators
     * @param _o1 the first objective vector
     * @param _o2 the second objective vector
     * @param _obj the objective in consideration
     * @return an epsilon indicator between two objective vectors
     */
    virtual double epsilon(const ObjectiveVector & _o1, const ObjectiveVector & _o2, const unsigned int _obj)=0;

};

#endif /*MOEOVECVSVECEPSILONBINARYMETRIC_H_*/
