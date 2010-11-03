/*
 <moCudaKswapEval.h>
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

#ifndef moCudaKswapEval_H
#define moCudaKswapEval_H
#include <eval/moCudaEval.h>
#include <eval/moCudakernelEval.h>

/**
 * class for the cuda evaluation
 */

template<class Neighbor, class IncrementEval>
class moCudaKswapEval: public moCudaEval<Neighbor> {

public:

	/**
	 * Define type of a solution corresponding to Neighbor
	 */
	typedef typename Neighbor::EOT EOT;
	/**
	 * Define type of a vector corresponding to Solution
	 */
	typedef typename EOT::ElemType T;
	/**
	 * Define type of a fitness corresponding to Solution
	 */
	typedef typename EOT::Fitness Fitness;

	using moCudaEval<Neighbor>::neighborhoodSize;
	using moCudaEval<Neighbor>::host_FitnessArray;
	using moCudaEval<Neighbor>::device_FitnessArray;
	using moCudaEval<Neighbor>::device_solution;
	using moCudaEval<Neighbor>::kernel_Dim;

	/**
	 * Constructor
	 * @param _neighborhoodSize the size of the neighborhood
	 * @param _incrEval the incremental evaluation
	 */

	moCudaKswapEval(unsigned int _neighborhoodSize, IncrementEval & _incrEval) :
		moCudaEval<Neighbor> (_neighborhoodSize), incrEval(_incrEval) {
	}

	/**
	 * Compute fitness for all solution neighbors in device
	 * @param _sol the solution which generate the neighborhood
	 */

	virtual void neighborhoodEval(EOT & _sol) {
	}

	/**
	 * Compute fitness for all solution neighbors in device
	 * @param _sol the solution which generate the neighborhood
	 * @param _mapping the array of indices mapping
	 * @param _Kswap the number of swap
	 */

	void neighborhoodEval(EOT & _sol, unsigned * _mapping, unsigned _Kswap) {

		// the solution vector size
		unsigned _size = _sol.size();

		// Get Current solution fitness
		Fitness fitness = _sol.fitness();

		//Allocate the space for solution in the  global memory of device
		cudaMalloc((void**) &device_solution.vect, _size * sizeof(T));

		//Copy the solution vector from the host to device
		cudaMemcpy(device_solution.vect, _sol.vect, _size * sizeof(T),
				cudaMemcpyHostToDevice);

		//Launch the Kernel to compute all neighbors fitness
		kernelKswap<EOT,Fitness,Neighbor,IncrementEval><<<kernel_Dim,BLOCK_SIZE >>>(incrEval,device_solution,device_FitnessArray,fitness,neighborhoodSize,_mapping,_Kswap);
		//Copy the result from device to host
		cudaMemcpy(host_FitnessArray, device_FitnessArray, neighborhoodSize
				* sizeof(Fitness), cudaMemcpyDeviceToHost);
	}

protected:

	IncrementEval & incrEval;
};

#endif

