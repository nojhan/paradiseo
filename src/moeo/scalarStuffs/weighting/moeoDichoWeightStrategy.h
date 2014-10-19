/*
 * <moeoDichoWeightStrategy.h>
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * Fran<-61><-89>ois Legillon
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

#ifndef MOEODICHOWEIGHTSTRAT_H_
#define MOEODICHOWEIGHTSTRAT_H_
#include "moeoVariableWeightStrategy.h"
#include <algorithm>
#include <utility>
#include "../../../eo/utils/rnd_generators.h"

/**
 * Change all weights according to a pattern ressembling to a "double strategy" 2 to 1 then 1 to 2.
 * Can only be applied to 2 objectives vector problem
 */
template <class MOEOT>
class moeoDichoWeightStrategy: public moeoVariableWeightStrategy<MOEOT>
{
	public:
		/**
		 * default constructor
		 */
		moeoDichoWeightStrategy():random(default_random),num(0){}

		/**
		 * constructor with a given random generator, for algorithms wanting to keep the same generator for some reason
		 * @param _random an uniform random generator
		 */
		moeoDichoWeightStrategy(UF_random_generator<double> &_random):random(_random),num(0){}

		/**
		 *
		 * @param _weights the weights to change
		 * @param moeot a moeot, will be kept in an archive in order to calculate weights later
		 */
		void operator()(std::vector<double> &_weights,const MOEOT &moeot){
			std::vector<double> res;
			ObjectiveVector tmp;
			_weights.resize(moeot.objectiveVector().size());
			if (arch.size()<2){
				//archive too small, we generate starting weights to populate it
				//if no better solution is provided, we will toggle between (0,1) and (1,0)
				arch(moeot);
				if (num==0){
					_weights[0]=0;
					_weights[1]=1;
					num++;
				}else{
					_weights[1]=0;
					_weights[0]=1;
					num=0;
					std::sort(arch.begin(),arch.end(),cmpParetoSort());
					it=arch.begin();
				}
				return;
			}else{
				if (it!=arch.end()){
					tmp=(*it).objectiveVector();
					it++;
					if (it==arch.end()){
						//we were at the last elements, recurse to update the archive
						operator()(_weights,moeot);
						return;
					}
					toAdd.push_back(moeot);
					res=normal(tmp,(*it).objectiveVector());
					_weights[0]=res[0];
					_weights[1]=res[1];
				}else{
					//we only add new elements to the archive once we have done an entire cycle on it,
					//to prevent iterator breaking
					//then we reset the iterator, and we recurse to start over
					arch(toAdd);
					toAdd.clear();
					std::sort(arch.begin(),arch.end(),cmpParetoSort());
					it=arch.begin();
					operator()(_weights,moeot);
					return;
				}
			}

		}



	private:
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;

		std::vector<double> normal(const ObjectiveVector &_obj1, const ObjectiveVector &_obj2){
			std::vector<double> res;
			double sum=0;
			for (unsigned int i=0;i<_obj1.size();i++){
				if (_obj1[i]>_obj2[i])
					res.push_back(_obj1[i]-_obj2[i]);
				else
					res.push_back(_obj2[i]-_obj1[i]);
				sum+=res[i];
			}
			for (unsigned int i=0;i<_obj1.size();i++) res[i]=res[i]/sum;
			return res;
		}
		struct cmpParetoSort
		{
			//since we apply it to a 2dimension pareto front, we can sort every objectiveVector
			// following either objective without problem
			bool operator()(const MOEOT & a,const MOEOT & b) const
			{
				return b.objectiveVector()[0]<a.objectiveVector()[0];
			}
		};

		UF_random_generator<double> &random;
		UF_random_generator<double> default_random;
		int num;
		moeoUnboundedArchive<MOEOT> arch;
		eoPop<MOEOT> toAdd;
		typename eoPop<MOEOT>::iterator it;
};

#endif
