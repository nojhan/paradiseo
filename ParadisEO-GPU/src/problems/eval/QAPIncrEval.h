/*
 <QAPIncrEval.h>
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

#ifndef __QAPIncrEval_H
#define __QAPIncrEval_H

#include <eval/moCudaEvalFunc.h>

/**
 * Incremental Evaluation of QAP
 */

template<class Neighbor>
class QAPIncrEval: public moCudaEvalFunc<Neighbor> {

public:

	typedef typename Neighbor::EOT EOT;
	typedef typename EOT::Fitness Fitness;

	/**
	 * Constructor
	 */

	QAPIncrEval() {
	}

	/**
	 * Destructor
	 */

	~QAPIncrEval() {
	}

	/*functor of  incremental evaluation of the solution(function inline can be called from host or device)
	 * @param _sol the solution to evaluate
	 * @param _fitness the fitness of the current solution
	 * @param _index the set of information to compute fitness neighbor
	 */

inline __host__ __device__ Fitness operator() (EOT & _sol,Fitness _fitness, unsigned int *_index) {

	Fitness tmp;

	/* dev_a & dev_b are global device variable, data specific to QAP problem (flow & distance matices)
	 * _index[0] the first position of swap
	 * _index[1] the second position of swap
	 * _index[2] the solution size
	 * _index[3] the id of neighbor
	 */

	tmp=_fitness+compute_delta(dev_a,dev_b,_sol,_index[0],_index[1],_index[2],_index[3]);
	return tmp;

}

    /* compute the new fitness of the solution after permutation of pair(i,j)(function inline called from host  device)
	 * @param _a the flow matrix of size*size (specific data of QAP problem must be declared as global device variable)
	 * @param _b the distance matrix of size*size (specific data of QAP problem must be declared as global device variable)
	 * @param _sol the solution to evaluate
	 * @param _i the first position of swap
	 * @param _j the second position of swap
	 * @param _size the id of neighbor
	 * @param _id the neighbor identifier
	 */

inline __host__ __device__  int compute_delta(int * _a,int * _b,EOT & _sol, int _i, int _j,int _size, int _id){

  int d;
  int k;

  d = (_a[_i*_size+_i]-_a[_j*_size+_j])*(_b[_sol[_j+_id*_size]*_size+_sol[_j+_id*_size]]-_b[_sol[_i+_id*_size]*_size+_sol[_i+_id*_size]]) +
    (_a[_i*_size+_j]-_a[_j*_size+_i])*(_b[_sol[_j+_id*_size]*_size+_sol[_i+_id*_size]]-_b[_sol[_i+_id*_size]*_size+_sol[_j+_id*_size]]);


  for (k = 0; k < _size; k=k+1)
    if (k!=_i && k!=_j)

      d = d + (_a[k*_size+_i]-_a[k*_size+_j])*(_b[_sol[k+_id*_size]*_size+_sol[_j+_id*_size]]-_b[_sol[k+_id*_size]*_size+_sol[_i+_id*_size]]) +
      (_a[_i*_size+k]-_a[_j*_size+k])*(_b[_sol[_j+_id*_size]*_size+_sol[k+_id*_size]]-_b[_sol[_i+_id*_size]*_size+_sol[k+_id*_size]]);

  return(d);
}

};

#endif
