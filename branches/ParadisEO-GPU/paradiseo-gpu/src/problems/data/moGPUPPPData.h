/*
 <moGPUPPPData.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Boufaras Karima, Thé Van Luong

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

#ifndef _moGPUPPPData_H_
#define _moGPUPPPData_H_

#include <memory/moGPUSpecificData.h>

template<class ElemType>
class moGPUPPPData: public moGPUSpecificData {

public:

	using moGPUSpecificData::GPUObject;

	/**
	 * Default Constructor
	 */

	moGPUPPPData() :
		moGPUSpecificData() {

		//(*this).load();
	}

	/**
	 * Constructor by copy
	 * @param _pppData the specific data of PPP
	 */

	moGPUPPPData(const moGPUPPPData & _pppData) {

		a_h = new int[Md * Nd];
		H_h = new int[Nd];

		for (int i = 0; i < Md; i++)
			for (int j = 0; j < Nd; j++) {
				a_h[i * Nd + j] = _pppData.a_h[i * Nd + j];
			}
		for (int k = 0; k < Nd; k++) {
			H_h[k] = _pppData.H_h[k];
		}

		GPUObject.memCopy(a_d, a_h, Nd * Md);
		GPUObject.memCopy(H_d, H_h, Nd);

	}

	/**
	 * Assignement operator
	 * @param _pppData the specific data of PPP
	 * @return a PPP Data
	 */

	moGPUPPPData & operator=(const moGPUPPPData & _pppData) {

		a_h = new int[Md * Nd];
		H_h = new int[Nd];
		for (int i = 0; i < Md; i++)
			for (int j = 0; j < Nd; j++) {
				a_h[i * Nd + j] = _pppData.a_h[i * Nd + j];
			}
		for (int k = 0; k < Nd; k++) {
			H_h[k] = _pppData.H_h[k];
		}

		GPUObject.memCopy(a_d, a_h, Nd * Md);
		GPUObject.memCopy(H_d, H_h, Nd);

		return (*this);
	}

	/*
	 * Destructor
	 */

	~moGPUPPPData() {
		GPUObject.memFree(a_d);
		GPUObject.memFree(H_d);
		delete[] a_h;
		delete[] H_h;
	}

	/*
	 *Load PPP data 
	 */

	void load(char * _fileName) {
	}

	void load() {

		int *v = new int[Nd];
		int *s = new int[Md];
		a_h = new int[Md * Nd];
		H_h = new int[Nd];
		for (int i = 0; i < Nd; i++)
					H_h[i] = 0;
		for (int i = 0; i < Md; i++) {
			for (int j = 0; j < Nd; j++) {
				a_h[i * Nd + j] =pow(-1,rand());
			}
		}
		for (int i = 0; i < Nd; i++) {
			v[i]=pow(-1,rand());
		}

		for (int i = 0; i < Md; i++) {
			s[i] = 0;
			for (int j = 0; j < Nd; j++)
				s[i] += a_h[i * Nd + j] * v[j];
			if (s[i] < 0) {
				for (int k = 0; k < Nd; k++)
					a_h[i * Nd + k] = -a_h[i * Nd + k];
				s[i] = -s[i];
			}
			if(s[i]>0)
				H_h[s[i]-1]++;
		}

		//Allocate and copy QAP data from CPU memory to GPU global memory
		GPUObject.memCopy(a_d, a_h, Nd * Md);
		GPUObject.memCopy(H_d, H_h, Nd);

		delete[] v;
		delete[] s;

	}


public:

	ElemType* a_h;
	ElemType* H_h;
	ElemType* a_d;
	ElemType* H_d;

};
#endif

