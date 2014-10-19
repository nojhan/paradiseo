/*
* <moeoVecVsVecAdditiveEpsilonBinaryMetric.h>
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

#ifndef MOEOVECVSVECADDITIVEEPSILONBINARYMETRIC_H_
#define MOEOVECVSVECADDITIVEEPSILONBINARYMETRIC_H_

#include "../comparator/moeoParetoObjectiveVectorComparator.h"
#include "moeoMetric.h"
#include "moeoVecVsVecEpsilonBinaryMetric.h"

/**
 * moeoVecVsVecAdditiveEpsilonBinaryMetric is the implementation of moeoVecVsVecEpsilonBinaryMetric whose calculate an additive epsilon indicator
 */
template < class ObjectiveVector >
class moeoVecVsVecAdditiveEpsilonBinaryMetric : public moeoVecVsVecEpsilonBinaryMetric < ObjectiveVector >
{

public:

    /**
     * Default Constructor: inherit of moeoVecVsVecEpsilonBinaryMetric
     */
    moeoVecVsVecAdditiveEpsilonBinaryMetric(bool _normalize=true): moeoVecVsVecEpsilonBinaryMetric < ObjectiveVector >(_normalize){}

private:


    /**
     * compute the additive epsilon indicator. Ieps+(A,B) equals the minimum factor eps such that any objective vector in B is eps-dominated by at least one objective vector in A.
     * a vector z1 is said to eps+-dominate another vector z2, if we can add each objective value in z2 by eps and the resulting objective vector is still weakly dominates by z1.
     * @param _o1 the first objective vector (correspond to A, must not have dominated elements)
     * @param _o2 the second objective vector (correspond to B, must not have dominated elements)
     * @param _obj the objective in consideration
     * @return the additive epsilon indicator between the two objective vector _o1 and _o2
     */
    double epsilon(const ObjectiveVector & _o1, const ObjectiveVector & _o2, const unsigned int _obj){

        double result;
        // if the objective _obj have to be minimized
        if (ObjectiveVector::Traits::minimizing(_obj))
        {
            // _o1[_obj] - _o2[_obj]
            result = ( (_o1[_obj] - moeoVecVsVecEpsilonBinaryMetric < ObjectiveVector >::bounds[_obj].minimum()) / moeoVecVsVecEpsilonBinaryMetric < ObjectiveVector >::bounds[_obj].range() ) - ( (_o2[_obj] - moeoVecVsVecEpsilonBinaryMetric < ObjectiveVector >::bounds[_obj].minimum()) / moeoVecVsVecEpsilonBinaryMetric < ObjectiveVector >::bounds[_obj].range() );
        }
        // if the objective _obj have to be maximized
        else
        {
            // _o2[_obj] - _o1[_obj]
            result = ( (_o2[_obj] - moeoVecVsVecEpsilonBinaryMetric < ObjectiveVector >::bounds[_obj].minimum()) / moeoVecVsVecEpsilonBinaryMetric < ObjectiveVector >::bounds[_obj].range() ) - ( (_o1[_obj] - moeoVecVsVecEpsilonBinaryMetric < ObjectiveVector >::bounds[_obj].minimum()) / moeoVecVsVecEpsilonBinaryMetric < ObjectiveVector >::bounds[_obj].range() );
        }
        return result;
    }



};

#endif /*MOEOVECVSVECEPSILONBINARYMETRIC_H_*/
