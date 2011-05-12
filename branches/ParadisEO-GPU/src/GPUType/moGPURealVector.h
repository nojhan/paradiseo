/*
 <moGPURealVector.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Karima Boufaras, Thé Van LUONG

 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  use,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".

 As a counterpart to the access to the source code and  rights to copy,
 modify and redistribute granted by the license, users are provided only
 with a limited warranty  and the software's author,  the holder of the
 economic rights,  and the successive licensors  have only  limited liability.

 In this respect, the user's attention is drawn to the risks associated
 with loading,  using,  modifying and/or developing or reproducing the
 software by the user in light of its specific status of free software,
 that may mean  that it is complicated to manipulate,  and  that  also
 therefore means  that it is reserved for developers  and  experienced
 professionals having in-depth computer knowledge. Users are therefore
 encouraged to load and test the software's suitability as regards their
 requirements in conditions enabling the security of their systems and/or
 data to be ensured and,  more generally, to use and operate it in the
 same conditions as regards security.
 The fact that you are presently reading this means that you have had
 knowledge of the CeCILL license and that you accept its terms.

 ParadisEO WebSite : http://paradiseo.gforge.inria.fr
 Contact: paradiseo-help@lists.gforge.inria.fr
 */

#ifndef __moGPURealVector_H_
#define __moGPURealVector_H_

#include <GPUType/moGPUVector.h>
#include <stdlib.h>

/**
 * Implementation of real vector representation on GPU.
 */

template<class Fitness>

class moGPURealVector: public moGPUVector<float, Fitness> {

public:

	using moGPUVector<float, Fitness>::vect;
	using moGPUVector<float, Fitness>::N;

	/**
	 * Default constructor.
	 */

	moGPURealVector() :
		moGPUVector<float, Fitness> () {

	}

	/**
	 *Constructor.
	 *@param _neighborhoodSize The neighborhood size.
	 */

	moGPURealVector(unsigned _neighborhoodSize) :
		moGPUVector<float, Fitness> (_neighborhoodSize) {
		create();
	}


	/**
	 *Initializer of random real vector.
	 */
	void create() {
		for (unsigned i = 0; i < N; i++)
			vect[i] = (float) rng.rand() / RAND_MAX;
	}

	/**
	 *Function inline to set the size of vector, called from host.
	 *@param _size the vector size
	 */

	virtual inline __host__ void setSize(unsigned _size) {

		if(_size<N) {
					moGPURealVector<Fitness> tmp_vect(_size);
					for (unsigned i = 0; i < tmp_vect.N; i++)
					tmp_vect.vect[i]= vect[i];
					(tmp_vect).invalidate();
					(*this)=tmp_vect;
				}
				else if(_size>N) {
					moGPURealVector<Fitness> tmp_vect(_size);
					for (unsigned i = 0; i <N; i++)
					tmp_vect.vect[i]= vect[i];
					(tmp_vect).invalidate();
					(*this)=tmp_vect;
				}


	}

	/**
	 * Write object. Called printOn since it prints the object _on_ a stream.
	 * @param _os A std::ostream.
	 */
	void printOn(std::ostream& os) const {
		EO<Fitness>::printOn(os);
		os << ' ';
		os << N << ' ';
		unsigned int i;
		for (i = 0; i < N; i++)
		os << vect[i] << ' ';

	}

};

#endif
