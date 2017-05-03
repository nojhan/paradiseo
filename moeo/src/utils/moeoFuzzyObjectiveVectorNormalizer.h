/*
  <moeoFuzzyObjectiveVectorNormalizer.h>
   Oumayma BAHRI

Author:
       Oumayma BAHRI <oumaymabahri.com>

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr	   

*/
//-----------------------------------------------------------------------------

#ifndef MOEOFUZZYOBJECTIVEVECTORNORMALIZER_H_
#define MOEOFUZZYOBJECTIVEVECTORNORMALIZER_H_
#include <eoPop.h>
#include <utils/eoRealBounds.h>
/**
  Adaptation of classical class "moeoObjectiveVectorNormalizer" to normalize fuzzy objective Vectors
 */
template <class MOEOT>
class moeoFuzzyObjectiveVectorNormalizer
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
		moeoFuzzyObjectiveVectorNormalizer(Scale _scale=make_dummy_scale(),Type max_param=100):scale(_scale),max(max_param)
	{}

		/**
		 * main fonction, normalize a triangular fuzzy vector defined with a triplet [first, second, third]. 
		 * @param _vec the vector
		 * @return the normalized vector
		 */
		virtual ObjectiveVector operator()(const ObjectiveVector &_vec){
			unsigned int dim=_vec.nObjectives();
			ObjectiveVector res;
			for (unsigned int i=0;i<dim;i++){
				res[i].first=(_vec[i].first-scale[i][1])*scale[i][0];
				res[i].second=(_vec[i].second-scale[i][1])*scale[i][0];
				res[i].third=(_vec[i].third-scale[i][1])*scale[i][0];
			}
			return res;
		}

		/**
		  normalize a population
		  @param pop the population to normalize
		  @return a vector of normalized Objective vectors
		 */

		static std::vector<ObjectiveVector> normalize(const eoPop<MOEOT> &pop, Type &max){
			moeoFuzzyObjectiveVectorNormalizer normalizer(pop,true, max);
			return normalizer(pop);
		}

		

	private:
		Scale scale;
		Type max;


};
#endif
