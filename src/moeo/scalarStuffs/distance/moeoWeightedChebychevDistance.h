/*
 * <moeoWeightedChebychevDistance.h>
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
#ifndef MOEOCHEBYCHEVDIST_H_
#define MOEOCHEBYCHEVDIST_H_


//#include <moeo>
#include <cmath>
/**
 * weighted chebychev distance
 */
template < class MOEOT>
class moeoWeightedChebychevDistance : public moeoObjSpaceDistance < MOEOT >
{
	public:
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename MOEOT::Fitness Fitness;

		/**
		 * constructor with a normalizer
		 * @param _rho
		 * @param _weight the weight to apply to each dimansion
		 * @param _normalizer the normalizer
		 */
		moeoWeightedChebychevDistance(unsigned int _rho, const ObjectiveVector& _weight, moeoObjectiveVectorNormalizer<MOEOT>& _normalizer): normalizer(_normalizer), weight(_weight), rho(_rho){}

		/**
		 * constructor with a normalizer
		 * @param _rho
		 * @param _weight the weight to apply to each dimansion
		 */
		moeoWeightedChebychevDistance(unsigned int _rho, const ObjectiveVector& _weight): normalizer(defaultNormalizer), weight(_weight), rho(_rho){}
		
		/**
		 * fonction which apply the metric to calculate a fitness
		 * @param _obj the point to evaluate
		 * @param _reference the reference to calculate the distance from
		 * @return the fitness conrresponding to the distance
		 */
		const Fitness operator()(const ObjectiveVector& _reference, const ObjectiveVector& _obj){
			unsigned int dim=_obj.size();
			Fitness res=0;
			ObjectiveVector obj_norm=normalizer(_obj);
			ObjectiveVector ref_norm=normalizer(_reference);
			for (unsigned i=0;i<dim;i++){
				res+=iteration(obj_norm,ref_norm,i);
			}
			return res;
		}

	private:
		moeoObjectiveVectorNormalizer<MOEOT> &normalizer;
		moeoObjectiveVectorNormalizer<MOEOT> defaultNormalizer;
		const ObjectiveVector &weight;
		double rho;

		Fitness iteration(const ObjectiveVector &obj,const ObjectiveVector &reference,int dim){
			Fitness res=abs(obj[dim]-reference[dim]);
			res=weight[dim]*pow(res,rho);
			res=pow(res,1/rho);
			return res;
		}
};
#endif
