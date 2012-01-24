/*
 <moGPUXChangeNeighborhood.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

 Karima Boufaras, Thé Van LUONG

 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  ue,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".

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

#ifndef __moGPUXChangeNeighborhood_h
#define __moGPUXChangeNeighborhood_h

#include <neighborhood/moGPUXChange.h>
#include <eval/moGPUEval.h>

template<class N>
class moGPUXChangeNeighborhood: public moGPUXChange<N> {

public:

	/**
	 * Define a Neighbor and type of a solution corresponding
	 */

	typedef N Neighbor;
	typedef typename Neighbor::EOT EOT;

	using moGPUXChange<Neighbor>::neighborhoodSize;
	using moGPUXChange<Neighbor>::currentIndex;
	using moGPUXChange<Neighbor>::indices;
	using moGPUXChange<Neighbor>::mapping;
	using moGPUXChange<Neighbor>::xChange;
	using moGPUXChange<Neighbor>::mutex;

	/**
	 * Constructor
	 * @param _neighborhoodSize the neighborhood size
	 * @param _xChange the number of x-change positions
	 */

	moGPUXChangeNeighborhood(unsigned int _neighborhoodSize,
			unsigned int _xChange) :
		moGPUXChange<Neighbor> (_neighborhoodSize, _xChange) {
		sendMapping = false;
		cudaMalloc((void**) &device_Mapping, sizeof(unsigned int)
				* neighborhoodSize * _xChange);
	}


	/**
	 *Destructor
	 */

	~moGPUXChangeNeighborhood() {

		cudaFree(device_Mapping);
	}

	/**
	 * Initialization of the neighborhood and mapping on device
	 * @param _solution the solution to explore
	 * @param _current the first neighbor
	 */

	virtual void init(EOT& _solution, Neighbor& _current) {

		moGPUXChange<Neighbor>::init(_solution, _current);
		if (!sendMapping) {
			cudaMemcpy(device_Mapping, mapping, xChange * neighborhoodSize
					* sizeof(unsigned int), cudaMemcpyHostToDevice);
			sendMapping = true;
		}
	}

	/**
	 * Return the class Name
	 * @return the class name as a std::string
	 */
	virtual std::string className() const {
		return "moGPUXChangeNeighborhood";
	}

protected:

	bool sendMapping;
	unsigned int * device_Mapping;

};

#endif
