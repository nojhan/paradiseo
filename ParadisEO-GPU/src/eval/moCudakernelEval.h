/*
 <moCudakernelEval.h>
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
#ifndef __moCudakernelEval_H
#define __moCudakernelEval_H


///////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * The kernel function called from the host and executed in device to compute all neighbors fitness at one time
 * @param _eval how to evaluate each neighbor
 * @param _solution representation of solution( vector of int,float....)
 * @param _allFitness Array of Fitness to save all neighbors fitness
 * @param _fitness the current solution fitness
 * @param _neighborhoodsize the size of the neighborhood
 */

template<class EOT, class Fitness, class Neighbor, class IncrementEval>

__global__ void kernelEval(IncrementEval _eval, EOT _solution, Fitness* _allFitness,
		Fitness _fitness, unsigned _neighborhoodsize) {

	// The thread identifier within a grid block's
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// array to save index to be changed
    unsigned int index[1];
	// In this representation each id identify one and only one neighbor in neighborhood
	if (id < _neighborhoodsize) {
	 //Change the id'th element of solution
     index[0]=id;
	 //Compute fitness for id'th neighbor
	 _allFitness[id] = _eval(_solution, _fitness,index);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * The kernel function called from the host and executed in device to compute all flip neighbors fitness at one time
 * @param _eval how to evaluate each neighbor
 * @param _solution representation of solution to flip
 * @param _allFitness Array of Fitness type to save all neighbors fitness
 * @param _fitness the current solution fitness
 * @param _neighborhoodsize the size of the neighborhood
 * @param _mapping the neighborhood mapping
 * @param _Kflip the number of bit to flip
 */

template<class EOT, class Fitness, class Neighbor, class IncrementEval>

__global__ void kernelKflip(IncrementEval _eval, EOT _solution, Fitness* _allFitness,
		Fitness _fitness, unsigned _neighborhoodsize, unsigned * _mapping,unsigned _Kflip) {

	// The thread identifier within a grid block's
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	//save temporary fitness
    unsigned tmp_fitness;
    //counter of number of flip to do
    unsigned i;
    // array to save index to be changed
    unsigned index[1];
	// In this representation each id identify one and only one neighbor in neighborhood
	if (id < _neighborhoodsize) {
	 //Init fitness with fitness of solution
     tmp_fitness=_fitness;
     //Evaluate neighbor after Kflip
     for(i=0;i<=_Kflip;i++){
         //The designed index to flip
    	 index[0]=_mapping[id + i * _neighborhoodsize];
    	 //Evaluate the neighbor
		 tmp_fitness= _eval(_solution, tmp_fitness, index);

     }
     //The final  fitness of the Id'th neighbor
		 _allFitness[id]=tmp_fitness;

}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * The kernel function called from the host and executed in device to compute all swap neighbors fitness at one time
 * @param _eval how to evaluate each neighbor
 * @param _solution representation ofsolution to swap
 * @param _sol_tmp to save temporary a solution element to swap
 * @param _allFitness Array of Fitness type to save all neighbors fitness
 * @param _fitness the current solution fitness
 * @param _neighborhoodsize the size of the neighborhood
 * @param _mapping the neighborhood mapping
 * @param _Kswap the number of swap to do
 * @param _size the solution size
 */

template<class EOT,class Fitness, class Neighbor, class IncrementEval>

__global__ void kernelKswap(IncrementEval _eval,EOT _solution ,EOT _sol_tmp, Fitness* _allFitness,
		Fitness _fitness, unsigned _neighborhoodsize, unsigned * _mapping,unsigned _Kswap,unsigned _size) {

	// The thread identifier within a grid block's
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	//save temporary fitness
	int tmp_fitness;
	//counter of number of swap to do
    unsigned i;
    // array to save index to be changed, solution size & thread id
    unsigned index[4];
	// In this representation each id identify one and only one neighbor in neighborhood
	if (id < _neighborhoodsize) {
		 //the first index to swap
    	 index[0]=_mapping[id];
    	 //the second index to swap
    	 index[1]=_mapping[id +_neighborhoodsize];
    	 //the solution size
    	 index[2]=_size;
    	 //the thread id
    	 index[3]=id;
    	 //Init the temporary fitness with the initial solution fitness
    	 tmp_fitness=_fitness;

    //Evaluate neighbor after K-swap
    for(i=2;i<=_Kswap+1;i++){
         //Evaluate neighbor with index case
    	 tmp_fitness=_eval(_solution, tmp_fitness, index);
    	 //Permut the solution
    	 _sol_tmp[id]=_solution[index[0]+id*index[2]];
		 _solution[index[0]+id*index[2]]=_solution[index[1]+id*index[2]];
		 _solution[index[1]+id*index[2]]=_sol_tmp[id];
		 //Init the next swap to do
	     index[0]=index[1];
		 index[1]=_mapping[id +i*_neighborhoodsize];

    }
      //save the final fitness of the id'th neighbor
     _allFitness[id]=tmp_fitness;
     }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * The kernel function called from the host and executed in device to compute all permutation neighbors fitness at one time
 * @param _eval how to evaluate each neighbor
 * @param _solution representation of  solution
 * @param _allFitness Array of Fitness type to save all neighbors fitness
 * @param _fitness the current solution fitness
 * @param _neighborhoodsize the size of the neighborhood
 * @param _mapping the neighborhood mapping
 * @param _size the solution size
 */

template<class EOT, class Fitness, class Neighbor, class IncrementEval>
__global__ void kernelPermutation(IncrementEval _eval, EOT _solution, Fitness* _allFitness,
		Fitness _fitness, unsigned _neighborhoodsize, unsigned * _mapping,unsigned _size) {

	// The thread identifier within a grid block's
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// array to save index to be changed, solution size
    unsigned index[4];
	// In this representation each id identify one and only one neighbor in neighborhood
	if (id < _neighborhoodsize) {
		 //The first index of permutation
    	 index[0]=_mapping[id];
    	 //The second index of permutation
    	 index[1]=_mapping[id +_neighborhoodsize];
    	 //The solution size
    	 index[2]=_size;
    	 //Puch 0 in the 3 index
    	 index[3]=0;
    	 _allFitness[id]=_eval(_solution,_fitness,index);
     }
}

#endif
