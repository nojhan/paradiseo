/*
* <moeoAugmentedQexploreWeightStrategy.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
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

#ifndef MOEOAUGMENTEDQEXPLWEIGHTSTRAT_H_
#define MOEOAUGMENTEDQEXPLWEIGHTSTRAT_H_
#include "moeoVariableWeightStrategy.h"
#include <algorithm>
#include <utility>
#include "../../../eo/utils/rnd_generators.h"

/**
 * Change all weights according to a pattern ressembling to a "double strategy" 2 to 1 then 1 to 2.
 * Can only be applied to 2 objectives vector problem
 */
template <class MOEOT>
class moeoAugmentedQexploreWeightStrategy: public moeoVariableWeightStrategy<MOEOT>
{
	public:
		/**
		 * default constructor
		 */
		moeoAugmentedQexploreWeightStrategy():depth(0),num(0),reset(true){
			nums.resize(1,0);
		}
		/**
		 *
		 * @param _weights the weights to change
		 * @param moeot a moeot, not used
		 */
		void operator()(std::vector<double> &_weights,const MOEOT &moeot){
			int dim=moeot.objectiveVector().size();
			bool res=false;
			int max=dim-1;
			if (depth==0) do_reset();
			while (!res) {
				res=translate(dim,_weights);
				next_num(dim);
				if (nums[0]>max){
					do_reset();
				}
			}
		}

	private:

		void next_num(int dim){
			int max=dim-1;
			int idx=nums.size()-1;
			if (depth==0){
				do_reset();
			}else{
				idx=nums.size()-1;
				while(idx>0 && nums[idx]==max) idx--;
				int to_assign=nums[idx]+1;
				for (unsigned int i=idx;i<nums.size();i++){
					nums[i]=to_assign;
				}
			}
		}

		bool translate(int dim, std::vector<double> &_weights){
			_weights.clear();
			_weights.resize(dim,0);
			for (unsigned int i=0;i<nums.size();i++){
				_weights[nums[i]]++;
				if (depth>1 && _weights[nums[i]]==depth) {
					return false;
				}
			}

			bool accept_pow=false;
			bool accept_prim=false;
			for (unsigned int i=0;i<_weights.size();i++){
				if (accept_pow || (_weights[i]!=1 && !is2pow(_weights[i]))) {
					accept_pow=true;
				}
				if (accept_prim || (coprim(_weights[i],depth)))
					accept_prim=true;

				_weights[i]=(_weights[i]+0.0)/(0.0+depth);
			}
			return accept_prim && accept_pow;
		}

		void do_reset(){
			if (depth==0) depth=1;
			else depth=depth*2;
			nums.resize(depth);
			for (unsigned int i=0;i<nums.size();i++){
				nums[i]=0;
			}
			reset=false;
		}

		int next_prime(int old){
			int res=old;
			bool prim=true;
			do{
				res+=1;
				prim=true;
				for (unsigned int i=2;i<=sqrt(res);i++){
					if ((res%i)==0) prim=false;
				}
			}while (!prim);
			return res;
		}

		bool coprim(int a, int b){
			if (b==0){
				return a==1;
			}else {
				return coprim(b,a%b);
			}
		}

		bool is2pow(int a){
			if (a==1 || a==0) {
				return true;
			}
			else if ((a%2)!=0) {
				return false;
			}

			else {
				return is2pow(a/2);
			}
		}

		std::vector<int> nums;
		int depth,num;
		bool reset;

};

#endif
