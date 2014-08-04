/*
* <moeoObjectiveVectorNormalizer.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2009
*
* Legillon François
*
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

#ifndef MOEOOBJVECNORM_H_
#define MOEOOBJVECNORM_H_
#include "../../eo/eoPop.h"
#include "../../eo/utils/eoRealBounds.h"
/**
  class to normalize each dimension of objectiveVectors
 */
template <class MOEOT>
class moeoObjectiveVectorNormalizer
{
	public:
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename MOEOT::ObjectiveVector::Type Type;
		typedef typename std::vector<std::vector<Type> > Scale;
		typedef eoRealInterval Bounds;



		/**
		  constructor with a supplied scale, usefull if you tweak your scale
		  @param _scale the scale for noramlzation
		  @param max_param the returned values will be between 0 and max
		  */
		moeoObjectiveVectorNormalizer(Scale _scale=make_dummy_scale(),Type max_param=100):scale(_scale),max(max_param)
	{}
		/**
		  constructor to create a normalizer from a given population
		  @param _pop the population to analyse to create the scale
		  @param max_param the returned values will be between 0 and max
		  */
		moeoObjectiveVectorNormalizer(eoPop<MOEOT> &_pop, Type max_param=100):scale(make_scale_from_pop(_pop,max_param)),max(max_param)
	{}
		/**
		  constructor to create a normalizer with given boundaries
		  @param _boundaries the supplied vectors should have their values between thos boundaries
		  @param max_param the returned values will be between 0 and max
		 **/
		moeoObjectiveVectorNormalizer(std::vector<Bounds> &_boundaries, Type max_param=100):scale(make_scale_from_bounds(_boundaries,max_param)), max(max_param)
	{}
		/**
		  constructor to create a normalizer from bounds
		  @param _bounds the supplied vectors should have their value in those bounds
		  @param max_param the returned values will be between 0 and max
		 **/
		moeoObjectiveVectorNormalizer(Bounds &_bounds, Type max_param=100 ):scale(make_scale_from_bounds(_bounds,max_param)), max(max_param)
	{}
		/**
		  constructor to create a normalizer from a worst vector and a best vector
		  @param _worst the worst possible vector
		  @param _best the best possible vector
		  @param max_param the maximum value for returned objectives
		  */
		moeoObjectiveVectorNormalizer(const ObjectiveVector &_best,const ObjectiveVector &_worst, Type max_param=100 ):scale(make_scale_from_minmax(_best,_worst,max_param)), max(max_param)
	{}


		/**
		 * Creates a scale which can be used in conjonction with a normalizer
		 * @param _pop the population to analyse
		 * @param max_param worst vector is set to it
		 * @return a scale to use with the normalizer
		 */
		static Scale make_scale_from_pop(eoPop<MOEOT> &_pop, Type max_param=100){
			Scale res;
			if (_pop.empty()) {
				std::cout<<"makeScale in moeoObjectiveVEctorNormalizer.h: pop is empty"<<std::endl;
				return res;
			}
			unsigned int dim=_pop[0].objectiveVector().nObjectives();
			std::vector<Type> amps;
			std::vector<Type> mins;
			unsigned int num_amp_max=0;
			//recherche des min et du max, par dimension
			for (unsigned int i=0;i<dim;i++){
				Type min=_pop[0].objectiveVector()[i];
				Type max=_pop[0].objectiveVector()[i];
				unsigned int pop_size=_pop.size();
				for (unsigned int j=0;j<pop_size;j++){
					if (_pop[j].objectiveVector()[i]< min) min=_pop[j].objectiveVector()[i];
					if (_pop[j].objectiveVector()[i]> max) max=_pop[j].objectiveVector()[i];
				}
				amps.push_back(max-min);
				mins.push_back(min);
				if (max-min>amps[num_amp_max])
					num_amp_max=i;
			}
			Type amp_max=amps[num_amp_max];
			for (unsigned int i=0;i<dim;i++){
				std::vector<Type> coefs;
				if(!max_param){
					coefs.push_back(amps[i]==0?1:amp_max/amps[i]);
				}
				else{
					coefs.push_back(amps[i]==0?1:max_param/amps[i]);
				}

				coefs.push_back(mins[i]);
				res.push_back(coefs);
			}
			return res;
		}

		/**
		  create a scale from bounds
		  @param _boundaries the boundaries
		  @param max the maximum for returned values
		  @return a scale
		  */
		static Scale make_scale_from_bounds(const std::vector<Bounds> &_boundaries,Type max=100){
			Scale res;
			for (unsigned i=0;i<_boundaries.size();i++){
				std::vector<Type> coeff;
				coeff.push_back(max/(_boundaries[i].maximum()-_boundaries[i].minimum()));
				coeff.push_back(_boundaries[i].minimum());
				res.push_back(coeff);
			}
			return res;
		}

		/**
		  create a scale from bounds
		  @param bounds the bounds (the same for each dimension)
		  @param max the maximum for returned values
		  @return a scale
		  */
		static Scale make_scale_from_bounds(const Bounds &bounds,Type max=100){
			Scale res;
			unsigned int dim=MOEOT::ObjectiveVector::nObjectives();
			for (unsigned i=0;i<dim;i++){
				std::vector<Type> coeff;
				coeff.push_back(max/(bounds.maximum()-bounds.minimum()));
				coeff.push_back(bounds.minimum());
				res.push_back(coeff);
			}
			return res;
		}

		/**
		  create a scale from a point with minimums in each dimension, and a point with ther max in each dimension
		  @param best the point with all mins
		  @param worst the point with all maxs
		  @param max the maximum for returned values
		  @return a scale
		  */
		static Scale make_scale_from_minmax(const ObjectiveVector &best, const ObjectiveVector &worst,Type max=100){
			Scale res;
			for (unsigned i=0;i<worst.nObjectives();i++){
				std::vector<Type> coeff;
				coeff.push_back(max/(worst[i]-best[i]));
				coeff.push_back(best[i]);
				res.push_back(coeff);
			}
			return res;
		}

		/**
		  create a default scale that does nothing when applied
		  @return a dummy scale
		  */
		static Scale make_dummy_scale(){
			unsigned int dim=MOEOT::ObjectiveVector::nObjectives();
			Scale res;
			for (unsigned int i=0;i<dim;i++){
				std::vector<Type> coeff;
				coeff.push_back(1);
				coeff.push_back(0);
				res.push_back(coeff);
			}
			return res;
		}
		/**
		 * main fonction, normalize a vector. All objective returned vectors will be between 0 and max previously
		 * supplied, be carefull about a possible rounding error.
		 * @param _vec the vector
		 * @return the normalized vector
		 */
		virtual ObjectiveVector operator()(const ObjectiveVector &_vec){
			unsigned int dim=_vec.nObjectives();
			ObjectiveVector res;
			for (unsigned int i=0;i<dim;i++){
				res[i]=(_vec[i]-scale[i][1])*scale[i][0];
			}
			return res;
		}

		/**
		  normalize a population
		  @param pop the population to normalize
		  @return a vector of normalized Objective vectors
		 */
		std::vector<ObjectiveVector> operator()(const eoPop<MOEOT> &pop){
			std::vector<ObjectiveVector> res;
			for (unsigned int i=0;i<pop.size();i++){
				res.push_back(operator()(pop[i].objectiveVector()));
			}
			return res;
		}

		/**
		  fast(to use, not in complexity) function to normalize a population
		  @param pop the population to normalize
		  @param max the returned values will be between 0 and max
		  @return a vector of normalized Objective vectors < max
		 */
		static std::vector<ObjectiveVector> normalize(const eoPop<MOEOT> &pop, Type &max){
			moeoObjectiveVectorNormalizer normalizer(pop,true, max);
			return normalizer(pop);
		}

		/**
		  Change the scale according to a new pop. Should be called everytime pop is updated
		  @param pop population to analyse
		 */
		void update_by_pop(eoPop<MOEOT> pop){
			scale=make_scale_from_pop(pop,max);
		}

		/** change the scale with the worst point and the best point
		  @param _max the worst point
		  @param _min the best point
		  */
		void update_by_min_max(const ObjectiveVector &_min,const ObjectiveVector &_max){
			scale=make_scale_from_minmax(_min,_max,max);
		}

		/** change the scale according to given boundaries
		  @param boundaries a vector of bounds corresponding to the bounds in each dimension
		  */
		void update_by_bounds(const std::vector<Bounds> &boundaries){
			scale=make_scale_from_bounds(boundaries);
		}
		/** change the scale according to bounds,them same is used in each dimension
		  @param bounds bounds corresponding to the bounds in each dimension
		  */
		void update_by_bounds(const Bounds &bounds){
			scale=make_scale_from_bounds(bounds);
		}


		/**
		  updates the scale
		  @param _scale the new scale
		  */
		void update_scale(Scale _scale){
			scale=_scale;
		}

		/**
		  change the maximum returned by the normalizer (if the scale is adapted)
		  @param _max the maximum returned
		  */
		void change_max(Type _max){
			for (unsigned int i=0;i<scale.size();i++){
				if (max) scale[i][0]=scale[i][0]*_max/max;


			}
			max=_max;
		}

	private:
		Scale scale;
		Type max;


};
#endif
