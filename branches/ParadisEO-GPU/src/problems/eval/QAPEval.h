/*
 <QAPEval.h>
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

#ifndef __QAPEval_H
#define __QAPEval_H

#include <problems/data/QAPData.h>

template<class EOT, class ElemType = typename EOT::ElemType>
class QAPEval: public eoEvalFunc<EOT> {

public:

	/**
	 * Constructor
	 * @param _qapData the specific data problem useful to evalute solution(flow & distance matrices of QAP problem)
	 */

	QAPEval(QAPData<ElemType> & _qapData) {
		qapData = _qapData;
	}

	/**
	 * Destructor
	 */

	~QAPEval() {
	}

	/**
	 * Full evaluation of the solution
	 * @param _sol the solution to evaluate
	 */

	void operator()(EOT & _sol) {
		int cost = 0;
		unsigned int size = qapData.getSize();
		for (unsigned int i = 0; i < size; i++)
			for (unsigned int j = 0; j < size; j++) {
				cost += qapData.a_h[i * size + j] * qapData.b_h[_sol[i] * size
						+ _sol[j]];
			}

		_sol.fitness(cost);
	}

protected:

	QAPData<ElemType> qapData;

};

#endif
