/*
 <PPPEval.h>
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

#ifndef __PPPEval_H
#define __PPPEval_H

#include <problems/data/PPPData.h>

template<class EOT, class ElemType = typename EOT::ElemType>
class PPPEval: public eoEvalFunc<EOT> {

public:

	/**
	 * Constructor
	 * @param _pppData the specific data problem useful to evalute solution( vector of 1 & _1 for PPP)
	 */

	PPPEval(PPPData<ElemType> & _pppData) {
		pppData = _pppData;
	}

	/**
	 * Destructor
	 */

	~PPPEval() {
	}

	/**
	 * Full evaluation of the solution
	 * @param _sol the solution to evaluate
	 */

	void operator()(EOT & _sol) {

		int *H;
		int tmp;
		int tmp_1 = 0;
		int tmp_2 = 0;

		H = new int[Nd];

		for (int i = 0; i < Md; i++) {
			tmp = 0;
			for (int j = 0; j < Nd; j++) {
				tmp += pppData.a_h[i * Nd + j] * _sol[j];
			}

			tmp_1 += abs(tmp) - tmp;
			if (tmp > 0)
				H[tmp-1]++;
		}

		for (int j = 0; j < Nd; j++) {
			tmp_2 += abs(pppData.H_h[j] - H[j]);
		}

		_sol.fitness(ca * tmp_1 + cb * tmp_2);

		delete[] H;

	}

protected:

	PPPData<ElemType> pppData;

};

#endif

