/*
 <moGPUSolType2Vector.h>
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

#ifndef _moGPUSolType2Vector_H_
#define _moGPUSolType2Vector_H_

#include <GPUType/moGPUVector.h>
#include <problems/types/moGPUCustomizeType.h>

/**
 * An Example of a customized vector representation on GPU.
 */

typedef struct sol2Type<int, float> ElemType;
template<class Fitness>
class moGPUSolType2Vector: public moGPUVector<ElemType, Fitness> {

public:
	/**
	 * Define vector type of vector corresponding to Solution
	 */

	using moGPUVector<ElemType, Fitness>::vect;
	using moGPUVector<ElemType, Fitness>::N;

	/**
	 * Default constructor.
	 */

	moGPUSolType2Vector() :
		moGPUVector<ElemType, Fitness> () {
	}

	/**
	 *Constructor.
	 *@param _size The size of the vector to create.
	 */

	moGPUSolType2Vector(unsigned _size) :
		moGPUVector<ElemType, Fitness> (_size) {
		create();
	}

	/**
	 *Assignment operator
	 *@param _vector The vector passed to the function determine the new content.
	 *@return a new vector.
	 */

	moGPUSolType2Vector<Fitness> & operator=(
			const moGPUSolType2Vector<Fitness> & _vector) {

		vect[0] = _vector[0];
		if (!(_vector.invalid()))
			fitness(_vector.fitness());
		else
			(*this).invalidate();
		return (*this);
	}

	/**
	 *How to fill the vector.
	 */

	virtual void create() {

		for (int i = 0; i < vect[0].size(); i++) {
			vect[0].tab1[i] = (int) (rng.rand() % (vect[0].size() - i) + i);
			vect[0].tab2[i] = (float) (rng.rand() % (vect[0].size() - i) + i);
		}
	}

	/**
	 *Function inline to set the size of vector, called from host and device.
	 *@param _size the vector size
	 */

	virtual void setSize(unsigned _size) {
		N = _size;
	}
	/**
	 * Print the solution
	 */

	virtual void printOn(std::ostream& os) const {

		EO<Fitness>::printOn(os);
		os << ' ';
		os << vect[0].size() << ' ';
		unsigned int i;
		for (i = 0; i < vect[0].size(); i++) {
			os << vect[0].tab1[i] << ' ';
		}
		os << endl;
		for (i = 0; i < vect[0].size(); i++) {
			os << vect[0].tab2[i] << ' ';
		}
		os << endl;

	}

inline __host__ __device__ unsigned size() {

	return N;

}

};

#endif
