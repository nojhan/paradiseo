/*
 * <moeoDistanceMetric.h>
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * Jeremie Humeau
 * Arnaud Liefooghe
 * François Legillon
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
#ifndef MOEODISTANCEMETRIC_H_
#define MOEODISTANCEMETRIC_H_

#include <cmath>
#include "../distance/moeoObjSpaceDistance.h"

/**
  Adapter to use Distances as a metric
  */
template < class MOEOT>
class moeoDistanceMetric : public moeoUnaryMetric < typename MOEOT::ObjectiveVector , typename MOEOT::Fitness >
{
	public:
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename MOEOT::Fitness Fitness;

		/**
		  constructor with a normalizer
		  @param _distance the distance
		  @param _referencePoint the point from which we evaluate the distance
		  @param _normalizer the normalizer
		 */
		moeoDistanceMetric(moeoObjSpaceDistance<MOEOT> &_distance, const ObjectiveVector &_referencePoint,moeoObjectiveVectorNormalizer<MOEOT>& _normalizer): distance(_distance), reference(_referencePoint),normalizer(_normalizer){}

		/**
		  constructor with a dummy normalizer
		  @param _distance the distance
		  @param _referencePoint the point from which we evaluate the distance
		 */
		moeoDistanceMetric(moeoObjSpaceDistance<MOEOT> &_distance, const ObjectiveVector &_referencePoint): distance(_distance), reference(_referencePoint),normalizer(defaultNormalizer){}

		/**
		  fonction which apply the metric to calculate a fitness
		  @param _obj the point to evaluate
		  @return the fitness conrresponding to the distance
		  */
		Fitness operator()(ObjectiveVector _obj){
			return distance(normalizer(reference), normalizer(_obj));
		}

	private:
		moeoObjSpaceDistance<MOEOT>& distance;
		const ObjectiveVector& reference;
		moeoObjectiveVectorNormalizer<MOEOT> defaultNormalizer;
		moeoObjectiveVectorNormalizer<MOEOT>& normalizer;

};
#endif
