/*
 <moGPUEval.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

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

#ifndef moGPUEval_H
#define moGPUEval_H
#include <eval/moEval.h>

/**
 * Abstract class for evaluation on GPU
 */

template<class Neighbor>
class moGPUEval: public moEval<Neighbor> {

public:

	/**
	 * Define type of a solution corresponding to Neighbor
	 **/
	typedef typename Neighbor::EOT EOT;
	/**
	 * Define type of a fitness corresponding to Solution
	 **/
	typedef typename EOT::Fitness Fitness;
	/**
	 * Define type of a vector corresponding to Solution
	 */
	typedef typename EOT::ElemType T;

	/**
	 * Constructor
	 * @param _neighborhoodSize the size of the neighborhood
	 */

	moGPUEval(unsigned int _neighborhoodSize) {

		neighborhoodSize = _neighborhoodSize;
		host_FitnessArray = new Fitness[neighborhoodSize];
		cudaMalloc((void**) &device_FitnessArray, neighborhoodSize
				* sizeof(Fitness));
		mutex = false;
#ifdef BLOCK_SIZE
		NEW_kernel_Dim = neighborhoodSize / BLOCK_SIZE + ((neighborhoodSize
						% BLOCK_SIZE == 0) ? 0 : 1);
		NEW_BLOCK_SIZE = BLOCK_SIZE;
#endif

	}

	/**
	 * Destructor
	 */

	~moGPUEval() {

		delete[] host_FitnessArray;
		cudaFree(device_FitnessArray);
		cudaFree(&device_solution);

	}

	/**
	 * Set fitness of a solution neighbors
	 *@param _sol the solution which generate the neighborhood
	 *@param _neighbor the current neighbor
	 */

	void operator()(EOT & _sol, Neighbor & _neighbor) {

		_neighbor.fitness(host_FitnessArray[_neighbor.index()]);

	}

	/**
	 * Compute fitness for all solution neighbors in device
	 * @param _sol the solution which generate the neighborhood
	 * @param _cpySolution Launch kernel with local copy option of solution in each thread
	 * @param _withCalibration an automatic configuration of kernel to launch( NB_THREAD BY BLOCK & NB_BLOCK BY KERNEL ),default (1)
	 */

	virtual void neighborhoodEval(EOT & _sol, bool _cpySolution,
			bool _withCalibration) {
	}

	/**
	 * Compute fitness for all solution neighbors in device
	 * @param _sol the solution which generate the neighborhood
	 * @param _mapping the associated neighborhood mapping
	 * @param _cpySolution Launch kernel with local copy option of solution in each thread
	 * @param _withCalibration an automatic configuration of kernel to launch( NB_THREAD BY BLOCK & NB_BLOCK BY KERNEL ), default (1)
	 */

	virtual void neighborhoodEval(EOT & _sol, unsigned int * _mapping,
			bool _cpySolution, bool _withCalibration) {
	}

protected:

	//the host array to save all neighbors fitness
	Fitness * host_FitnessArray;
	//the device array to save neighbors fitness computed in device
	Fitness * device_FitnessArray;
	//the device solution
	EOT device_solution;
	//the size of neighborhood
	unsigned int neighborhoodSize;
	//Assure mapping transfer from CPU memory to GPU global memory for once time
	unsigned mutex;
	//GPU kernel dimension
	unsigned NEW_BLOCK_SIZE;
	unsigned NEW_kernel_Dim;

};

#endif
