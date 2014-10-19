/*
 * <moeoAugmentedAchievmentScalarizingFunctionDistance.h>
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * Jeremie Humeau
 * Arnaud Liefooghe
 * Legillon François
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
#ifndef MOEOASFAORDIST_H_
#define MOEOASFAORDIST_H_


//#include <moeo>
#include <cmath>

/**
  Order representing Achievment scalarizing function aproach to calculate a metric
 */
template < class MOEOT>
class moeoAugmentedAchievementScalarizingFunctionDistance : public moeoObjSpaceDistance< MOEOT >
{
	public:
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename MOEOT::Fitness Fitness;
		/**
		  constructor with a normalizer
		  @param _rho
		  @param _weight the weight to apply to each dimansion
		  @param _normalizer the normalizer
		 */
		moeoAugmentedAchievementScalarizingFunctionDistance(unsigned int _rho,const ObjectiveVector &_weight,moeoObjectiveVectorNormalizer<MOEOT> &_normalizer): normalizer(_normalizer),weight(_weight),rho(_rho)
	{}
		/**
		  constructor without a normalizer
		  @param _rho
		  @param _weight the weight to apply to each dimansion
		 */
		moeoAugmentedAchievementScalarizingFunctionDistance(unsigned int _rho,const ObjectiveVector &_weight): normalizer(defaultNormalizer),weight(_weight),rho(_rho)
	{}


		/**
		  fonction which apply the metric to calculate a fitness
		  @param _reference the reference point to calculate the distance
		  @param _obj the point to evaluate
		  @return the fitness conrresponding to the distance
		 */
		const Fitness operator()(const ObjectiveVector &_reference,const ObjectiveVector &_obj){
			unsigned int dim=_obj.size();
			Fitness res=0;
			Fitness max=iteration(_obj,_reference,0,_obj.minimizing(0));
			for (unsigned i=0;i<dim;i++){
				Fitness tmp=iteration(_obj,_reference,i,_obj.minimizing(i));
				if (max<tmp) 
					max=tmp;
				res+=tmp;
			}
			return res+rho*max;
		}

	private:
		moeoObjectiveVectorNormalizer<MOEOT> defaultNormalizer;
		moeoObjectiveVectorNormalizer<MOEOT> &normalizer;
		const ObjectiveVector &weight;
		double rho;
		Fitness iteration(const ObjectiveVector &obj,const ObjectiveVector &reference,int dim,bool mini){
			ObjectiveVector obj_norm=normalizer(obj);
			ObjectiveVector ref_norm=normalizer(reference);
			Fitness res;
			if (mini){
				res=obj_norm[dim]-ref_norm[dim];
			}else{

				res=ref_norm[dim]-obj_norm[dim];
			}
			res=weight[dim]*res;
			return res;
		}
};
#endif
