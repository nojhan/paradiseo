/*
* <moeoFixedTimeOneDirectionWeightStrategy.h>
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

#ifndef MOEOFTODWSWEIGHTSTRAT_H_
#define MOEOFTODWSWEIGHTSTRAT_H_
#include "moeoVariableWeightStrategy.h"
#include <algorithm>
#include <utility>
#include "../../../eo/utils/rnd_generators.h"

/**
 * Change all weights according to a simple strategy by adding a step every generation
 */
template <class MOEOT>
class moeoFixedTimeOneDirectionWeightStrategy: public moeoVariableWeightStrategy<MOEOT>
{
	public:
		/**
		 * default constructor
		 * @param _step how much we want the weight to change every iteration
		 */
		moeoFixedTimeOneDirectionWeightStrategy(double _step):step(_step),current(0){}
		/**
		 *
		 * @param _weights the weights to change
		 * @param moeot a moeot, not used
		 */
		void operator()(std::vector<double> &_weights,const MOEOT &moeot){
			double res;
			if (current==1){
				res=1;
				current=0;
			}else res=current;
			if (current+step>1){
				current=1;
			}else current+=step;
			_weights[0]=res;
			_weights[1]=1-res;
		}

	private:
		double step;
		double current;
};

#endif
