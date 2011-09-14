/*
 <moGPUMappingEvalByCpy.h>
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

#ifndef __moGPUMappingEvalByCpy_H
#define __moGPUMappingEvalByCpy_H
#include <eval/moGPUEval.h>
#include <eval/moGPUMappingKernelEvalByCpy.h>
#include <performance/moGPUTimer.h>

/**
 * class for the Mapping neighborhood evaluation
 */

template<class Neighbor, class Eval>
class moGPUMappingEvalByCpy: public moGPUEval<Neighbor> {

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

	using moGPUEval<Neighbor>::neighborhoodSize;
	using moGPUEval<Neighbor>::host_FitnessArray;
	using moGPUEval<Neighbor>::device_FitnessArray;
	using moGPUEval<Neighbor>::device_solution;
	using moGPUEval<Neighbor>::NEW_BLOCK_SIZE;
	using moGPUEval<Neighbor>::NEW_kernel_Dim;
	using moGPUEval<Neighbor>::mutex;

	/**
	 * Constructor
	 * @param _neighborhoodSize the size of the neighborhood
	 * @param _eval how to evaluate a neighbor
	 */

	moGPUMappingEvalByCpy(unsigned int _neighborhoodSize, Eval & _eval) :
		moGPUEval<Neighbor> (_neighborhoodSize), eval(_eval) {
	}

	/**
	 * Destructor
	 */
	~moGPUMappingEvalByCpy() {
	}

	/**
	 * Compute fitness for all solution neighbors in device with associated mapping
	 * @param _sol the solution that generate the neighborhood to evaluate parallely
	 * @param _mapping the array of mapping indexes that associate a neighbor identifier to X-position
	 * @param _cpySolution Launch kernel with local copy option of solution in each thread if it's set to true
	 * @param _withCalibration an automatic kernel configuration, fix nbr of thread by block and nbr of grid by kernel
	 */
	void neighborhoodEval(EOT & _sol, unsigned int * _mapping,
			bool _cpySolution, bool _withCalibration) {
		if (_cpySolution) {
			unsigned size = _sol.size();
			// Get Current solution fitness
			Fitness fitness = _sol.fitness();
			if (!mutex) {
				//Allocate the space for solution in the device global memory
				cudaMalloc((void**) &device_solution.vect, size * sizeof(T));
				if (_withCalibration)
					calibration(_sol, _mapping);
				mutex = true;
			}
			//Copy the solution vector from the host to device
			cudaMemcpy(device_solution.vect, _sol.vect, size * sizeof(T),
					cudaMemcpyHostToDevice);
			//Launch the Kernel to compute all neighbors fitness,using a given mapping
			moGPUMappingKernelEvalByCpy<T,Fitness,Eval><<<NEW_kernel_Dim,NEW_BLOCK_SIZE >>>(eval,device_solution.vect,device_FitnessArray,fitness,_mapping,neighborhoodSize);
			cudaMemcpy(host_FitnessArray, device_FitnessArray, neighborhoodSize
					* sizeof(Fitness), cudaMemcpyDeviceToHost);

		} else
			cout << "It's evaluation by copy set cpySolution to true" << endl;
	}

	virtual void calibration(EOT & _sol, unsigned int * _mapping) {

		unsigned size = _sol.size();
		Fitness fitness = _sol.fitness();
		unsigned NB_THREAD[6] = { 16, 32, 64, 128, 256, 512 };
		double mean_time[7] = { 0, 0, 0, 0, 0, 0 };
		unsigned i = 0;
		double best_time = 0;
		unsigned tmp_kernel_Dim;
		best_time = RAND_MAX;
#ifndef BLOCK_SIZE

		do {
			tmp_kernel_Dim = neighborhoodSize / NB_THREAD[i]
					+ ((neighborhoodSize % NB_THREAD[i] == 0) ? 0 : 1);
			for (unsigned k = 0; k < 5; k++) {
				cudaMemcpy(device_solution.vect, _sol.vect, size * sizeof(T),
						cudaMemcpyHostToDevice);
				moGPUTimer timer;
				timer.start();
				moGPUMappingKernelEvalByCpy<T,Fitness,Eval><<<tmp_kernel_Dim,NB_THREAD[i]>>>(eval,device_solution.vect,device_FitnessArray,fitness,_mapping,neighborhoodSize);
				timer.stop();
				mean_time[i] += (timer.getTime());
				timer.deleteTimer();
			}
			if (best_time >= (mean_time[i] / 5)) {
				best_time = mean_time[i] / 5;
				NEW_BLOCK_SIZE = NB_THREAD[i];
				NEW_kernel_Dim = tmp_kernel_Dim;
			}
			i++;
		} while (i < 6);

#else

		tmp_kernel_Dim =NEW_kernel_Dim;
		for (unsigned k = 0; k < 5; k++) {
			cudaMemcpy(device_solution.vect, _sol.vect, size * sizeof(T),
					cudaMemcpyHostToDevice);
			moGPUTimer timer;
			timer.start();
			moGPUMappingKernelEvalByCpy<T,Fitness,Eval><<<tmp_kernel_Dim,BLOCK_SIZE>>>(eval,device_solution.vect,device_FitnessArray,fitness,_mapping,neighborhoodSize);
			timer.stop();
			mean_time[6] += (timer.getTime());
			timer.deleteTimer();
		}
		if (best_time >= (mean_time[6] / 5))
		best_time = mean_time[6] / 5;
		do {
			tmp_kernel_Dim = neighborhoodSize / NB_THREAD[i]
			+ ((neighborhoodSize % NB_THREAD[i] == 0) ? 0 : 1);
			for (unsigned k = 0; k < 5; k++) {
				cudaMemcpy(device_solution.vect, _sol.vect, size * sizeof(T),
						cudaMemcpyHostToDevice);
				moGPUTimer timer;
				timer.start();
				moGPUMappingKernelEvalByCpy<T,Fitness,Eval><<<tmp_kernel_Dim,NB_THREAD[i]>>>(eval,device_solution.vect,device_FitnessArray,fitness,_mapping,neighborhoodSize);
				timer.stop();
				mean_time[i] += (timer.getTime());
				timer.deleteTimer();
			}
			if (best_time >= (mean_time[i] / 5)) {
				best_time = mean_time[i] / 5;
				NEW_BLOCK_SIZE = NB_THREAD[i];
				NEW_kernel_Dim = tmp_kernel_Dim;
			}
			i++;
		}while (i < 6);

#endif

	}

protected:

	Eval & eval;

};

#endif
