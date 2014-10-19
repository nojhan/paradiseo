/*
* <moeoAnytimeWeightStrategy.h>
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

#ifndef MOEOANYTIMEWEIGHTSTRAT_H_
#define MOEOANYTIMEWEIGHTSTRAT_H_
#include "moeoVariableWeightStrategy.h"
#include <algorithm>
#include <utility>
#include "../../../eo/utils/rnd_generators.h"

/**
 * Change all weights according to a pattern ressembling to a "double strategy" 2 to 1 then 1 to 2.
 * Can only be applied to 2 objectives vector problem
 */
template <class MOEOT>
class moeoAnytimeWeightStrategy: public moeoVariableWeightStrategy<MOEOT>
{
	public:
		/**
		 * default constructor
		 */
		moeoAnytimeWeightStrategy():random(default_random),depth(0){}

		/**
		 * constructor with a given random generator, for algorithms wanting to keep the same generator for some reason
		 * @param _random an uniform random generator
		 */
		moeoAnytimeWeightStrategy(UF_random_generator<double> &_random):random(_random), depth(0){}

		/**
		 *
		 * @param _weights the weights to change
		 * @param _moeot not used
		 */
		void operator()(std::vector<double>& _weights, const MOEOT& _moeot){
			if (depth<2){
				if (depth==0) toTest.push_back(0.5);
				_weights[0]=depth;
				_weights[1]=1-_weights[0];
				depth++;
				old1=0;
				old2=1;
				return;
			}
			if (!toTest.empty()){
				_weights[0]=toTest.front();
				_weights[1]=1-_weights[0];
				toTest.erase(toTest.begin());
				toTest.push_back((_weights[0]+old1)/2);
				toTest.push_back((_weights[0]+old2)/2);
				old2=old1;
				old1=_weights[0];
			}else{
				std::cout<<"Error: Strange occurence in moeoAnytimeWeightStrategy "<<std::endl;
			}
		}

	private:
		double old1;
		double old2;
		UF_random_generator<double> &random;
		UF_random_generator<double> default_random;
		int depth;
		std::list<double> toTest;
};

#endif
