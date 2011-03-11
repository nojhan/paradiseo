/*
 <moKswapNeighborhood.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Boufaras Karima, Thé Van Luong

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

#ifndef _moCudaKswapNeighborhood_h
#define _moCudaKswapNeighborhood_h

#include <neighborhood/moKswapNeighborhood.h>
#include <eval/moCudaEval.h>

/**
 * K-flip Neighborhood
 */
template<class N>
class moCudaKflipNeighborhood: public moKswapNeighborhood<N> {

public:

	/**
	 * Define a Neighbor and type of a solution corresponding
	 */

	typedef N Neighbor;
	typedef typename Neighbor::EOT EOT;

	using moKswapNeighborhood<Neighbor>::neighborhoodSize;
	using moKswapNeighborhood<Neighbor>::currentIndex;
	using moKswapNeighborhood<Neighbor>::indices;
	using moKswapNeighborhood<Neighbor>::mapping;
	using moKswapNeighborhood<Neighbor>::Kswap;
	using moKswapNeighborhood<Neighbor>::size;
	using moKswapNeighborhood<Neighbor>::mutex;

	/**
	 * Constructor
	 * @param _size the size of solution
	 * @param _size the solution size
	 * @param _Kflip the number of bit to flip
	 * @param _eval show how to evaluate neighborhood of a solution at one time
	 */

	moCudaKflipNeighborhood(unsigned int _size, unsigned int _Kflip,
			moCudaEval<Neighbor>& _eval) :
		moKswapNeighborhood<Neighbor> (_size, _Kflip), eval(_eval) {
		sendMapping = true;
		cudaMalloc((void**) &device_Mapping, sizeof(unsigned int)
				* neighborhoodSize * (_Kflip + 1));
	}
	;

	/**
	 *Destructor
	 */

	~moCudaKflipNeighborhood() {

		cudaFree(device_Mapping);
	}

	/**
	 * Initialization of the neighborhood
	 * @param _solution the solution to explore
	 * @param _current the first neighbor
	 */

	virtual void init(EOT& _solution, Neighbor& _current) {

		moKswapNeighborhood<Neighbor>::init(_solution, _current);
		if (sendMapping) {
			cudaMemcpy(device_Mapping, mapping, (Kswap + 1) * neighborhoodSize
					* sizeof(unsigned int), cudaMemcpyHostToDevice);
			sendMapping = false;
		}
		//Compute all neighbors fitness at one time
		eval.neighborhoodKflipEval(_solution, device_Mapping, Kswap);
	}

	/**
	 * Return the class Name
	 * @return the class name as a std::string
	 */
	virtual std::string className() const {
		return "moCudaKSwapNeighborhood";
	}

protected:

	moCudaEval<Neighbor>& eval;
	bool sendMapping;
	unsigned int * device_Mapping;

};

#endif
