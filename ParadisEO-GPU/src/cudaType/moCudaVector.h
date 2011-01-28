/*
 <moCudaVector.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Thé Van LUONG, Karima Boufaras

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

#ifndef _moCudaVector_H_
#define _moCudaVector_H_

#include <eo>

/**
 * Implementation of vector representation on CUDA.
 */

template<class ElemT, class Fitness>

class moCudaVector: public EO<Fitness> {

public:

	typedef ElemT ElemType;
	using moCudaVector<int, Fitness>::vect;
	using moCudaVector<int, Fitness>::N;

	/**
	 * Default constructor.
	 */

	moCudaVector() :
		N(0) {
	}

	/**
	 *Constructor.
	 *@param _size The neighborhood size.
	 */

	moCudaVector(unsigned _size) {

		N = _size;

		vect = new ElemType[_size];

		create();
	}

	/**
	 * Destructor.
	 */

	~moCudaVector() {
		if (N > 1)
			delete[] vect;
	}

	/**
	 *How to fill the vector.
	 */

	virtual void create() {

	}

	/**
	 *Assignment operator
	 *@param _vector The vector passed to the function determine the new content.
	 *@return a new vector.
	 */

	virtual moCudaVector& operator=(const moCudaVector & _vector) {

		N = _vector.N;
		vect = new ElemType[N];
		for (unsigned i = 0; i < N; i++)
			vect[i] = _vector.vect[i];
		fitness(_vector.fitness());
		return (*this);

	}

	/**
	 *An accessor read only on the i'th element of the vector (function inline can be called from host or device).
	 *@param
	 *_i The i'th element of vector.
	 *@return
	 *The i'th element of the vector for read only
	 */

	inline __host__ __device__ const ElemType & operator[](unsigned _i) const {

		return vect[_i];

	}

	/**
	 *An accessor read-write on the i'th element of the vector(function inline can be called from host or device).
	 *@param _i The i'th element of the vector.
	 *@return The i'th element of the vector for read-write
	 */

	inline __host__ __device__ ElemType & operator[](unsigned _i) {

		return vect[_i];

	}

	/**
	 *Function inline to get the size of vector, called from host and device.
	 *@return The vector size's
	 */

	inline __host__ __device__ unsigned size() {

		return N;

	}

	virtual void printOn(std::ostream& os) const {
		EO<Fitness>::printOn(os);
		os << ' ';
		os << N << ' ';
		unsigned int i;
		for (i = 0; i < N; i++)
			os << vect[i] << ' ';

	}

};

#endif
