/*
* <moeoRandWeightStrategy.h>
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

#ifndef MOEORANDWEIGHTSTRAT_H_
#define MOEORANDWEIGHTSTRAT_H_
#include "moeoVariableWeightStrategy.h"

/**
 * Change all weights randomly.
 */
template <class MOEOT>
class moeoRandWeightStrategy: public moeoVariableWeightStrategy<MOEOT>
  {
	  public:
		  /**
		   * default constructor
		   */
		  moeoRandWeightStrategy():random(default_random){}

		  /**
		   * constructor with a given random generator, for algorithms wanting to keep the same generator for some reason
		   * @param _random an uniform random generator
		   */
		  moeoRandWeightStrategy(UF_random_generator<double> &_random):random(_random){}

		  /**
		   * main function, fill the weight randomly
		   * @param _weights the weights to change
		   * @param _moeot not used
		   */
		  void operator()(std::vector<double> &_weights,const MOEOT &_moeot){
			  double sum=0;
			  for (unsigned int i=0;i<_weights.size();i++){
				  double rnd=random(100000);
				  sum+=rnd;
				  _weights[i]=rnd;
			  }
			  //we divide by the sum in order to keep the weight sum equal to 1
			  for (unsigned int i=0;i<_weights.size();i++){
				  _weights[i]=_weights[i]/sum;
			  }
		  }

	  private:
		  UF_random_generator<double> &random;
		  UF_random_generator<double> default_random;
  };

#endif
