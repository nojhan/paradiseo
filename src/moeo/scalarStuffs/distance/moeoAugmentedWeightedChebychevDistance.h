/*
 * <moeoAugmentedWeightedChebychevDistance.h>
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * François Legillon
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
#ifndef MOEOCHEBYCHEVORDIST_H_
#define MOEOCHEBYCHEVORDIST_H_

#include "../../distance/moeoObjSpaceDistance.h"

/**
  order representing chebychev distance
  */
template < class MOEOT>
class moeoAugmentedWeightedChebychevDistance : public moeoObjSpaceDistance < MOEOT >
{
	public:
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename MOEOT::Fitness Fitness;

		/**
		  constructor with a normalizer
		  @param _rho
		  @param _weight the weight to apply to each dimension
		  @param _normalizer the normalizer
		 */
		moeoAugmentedWeightedChebychevDistance(unsigned int _rho, const ObjectiveVector& _weight, moeoObjectiveVectorNormalizer<MOEOT>& _normalizer): normalizer(_normalizer), weight(_weight), rho(_rho){}

		/**
		  constructor with a dummy normalizer
		  @param _rho
		  @param _weight the weight to apply to each dimension
		 */
		moeoAugmentedWeightedChebychevDistance(unsigned int _rho, const ObjectiveVector& _weight): normalizer(defaultNormalizer), weight(_weight), rho(_rho){}
		
		/**
		  fonction which calculate a fitness
		  @param _reference the reference to calculate the distance from
		  @param _obj the point to evaluate
		  @return the fitness conrresponding to the distance
		  */
		const Fitness operator()(const ObjectiveVector& _reference, const ObjectiveVector& _obj){
			unsigned int dim=_obj.size();
			Fitness res=iteration(_obj,_reference,0);
			Fitness max=res*weight[0];
			for (unsigned i=1;i<dim;i++){
				Fitness tmp=iteration(_obj,_reference,i);
				if (tmp*weight[i]>max)
					max=tmp*weight[i];
				res+=tmp;
			}
			res=res*rho+max;
			return res;
		}

	private:
		moeoObjectiveVectorNormalizer<MOEOT> &normalizer;
		moeoObjectiveVectorNormalizer<MOEOT> defaultNormalizer;
		const ObjectiveVector &weight;
		double rho;

		Fitness iteration(const ObjectiveVector &obj,const ObjectiveVector &reference,int dim){
			ObjectiveVector obj_norm=normalizer(obj);
			ObjectiveVector ref_norm=normalizer(reference);
			Fitness res=abs(obj_norm[dim]-ref_norm[dim]);
			return res;
		}
};
#endif
