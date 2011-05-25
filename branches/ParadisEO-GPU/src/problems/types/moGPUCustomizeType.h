/*
 <moGPUCustomizeType.h>
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

#ifndef _moGPUCustomizeType_H_
#define _moGPUCustomizeType_H_

/**
 * Implementation of an Example of customized type
 */

template<class T1, class T2>
typedef struct sol2Type {

	T1 tab1[SIZE];
	T2 tab2[SIZE];

inline __host__ __device__ sol2Type& operator=(const sol2Type _vector) {
	for (unsigned i = 0; i < SIZE; i++) {

		tab1[i] = _vector.tab1[i];
		tab2[i] = _vector.tab2[i];
	}
	return (*this);
}

inline __host__ __device__ unsigned size() {

	return SIZE;

}
}Type2;

template<class T1, class T2,class T3>
typedef struct sol3Type {

	T1 tab1[SIZE];
	T2 tab2[SIZE];
	T3 tab3[SIZE];

inline __host__ __device__ sol3Type& operator=(const sol3Type _vector) {

	for (unsigned i = 0; i < SIZE; i++) {

		tab1[i] = _vector.tab1[i];
		tab2[i] = _vector.tab2[i];
		tab3[i] = _vector.tab3[i];
	}
	return (*this);
}

inline __host__ __device__ unsigned size() {

	return SIZE;

}
}Type3;

template<class T1, class T2,class T3,class T4>
typedef struct sol4Type {

	T1 tab1[SIZE];
	T2 tab2[SIZE];
	T3 tab3[SIZE];
	T4 tab4[SIZE];

inline __host__ __device__ sol4Type& operator=(const sol4Type _vector) {

	for (unsigned i = 0; i < SIZE; i++) {
		tab1[i] = _vector.tab1[i];
		tab2[i] = _vector.tab2[i];
		tab3[i] = _vector.tab3[i];
		tab4[i] = _vector.tab4[i];
	}
	return (*this);
}

inline __host__ __device__ unsigned size() {

	return SIZE;

}
}Type4;

#endif
