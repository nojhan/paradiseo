/*
 <moGPUXChangeNeighborhoodByCpy.h>
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

#ifndef __moGPUXChangeNeighborhoodByCpy_h
#define __moGPUXChangeNeighborhoodByCpy_h

#include <neighborhood/moGPUXChangeNeighborhood.h>
#include <eval/moGPUEval.h>

template<class N>
class moGPUXChangeNeighborhoodByCpy: public moGPUXChangeNeighborhood<N> {

public:

	/**
	 * Define a Neighbor and type of a solution corresponding
	 */

	typedef N Neighbor;
	typedef typename Neighbor::EOT EOT;

	/*A tester*/
	using moGPUXChangeNeighborhood<Neighbor>::neighborhoodSize;
	using moGPUXChangeNeighborhood<Neighbor>::currentIndex;
	using moGPUXChangeNeighborhood<Neighbor>::indices;
	using moGPUXChangeNeighborhood<Neighbor>::mapping;
	using moGPUXChangeNeighborhood<Neighbor>::xChange;
	using moGPUXChangeNeighborhood<Neighbor>::mutex;
	using moGPUXChangeNeighborhood<Neighbor>::device_Mapping;

	/**
	 * Constructor
	 * @param _neighborhoodSize the neighborhood size
	 * @param _xChange the number of x-change positions
	 * @param _eval show how to evaluate neighborhood of a solution at one time
	 */

	moGPUXChangeNeighborhoodByCpy(unsigned int _neighborhoodSize,
			unsigned int _xChange,moGPUEval<Neighbor>& _eval) :
		moGPUXChangeNeighborhood<Neighbor> (_neighborhoodSize, _xChange), eval(_eval){
	}


	/**
	 * Initialization of the neighborhood
	 * @param _solution the solution to explore
	 * @param _current the first neighbor
	 */

	virtual void init(EOT& _solution, Neighbor& _current) {

		moGPUXChangeNeighborhood<Neighbor>::init(_solution, _current);
		//Compute all neighbors fitness at one time
		 eval.neighborhoodEval(_solution, device_Mapping,1,1);
	}

	/**
	 * Return the class Name
	 * @return the class name as a std::string
	 */
	virtual std::string className() const {
		return "moGPUXChangeNeighborhoodByCpy";
	}

protected:

	moGPUEval<Neighbor>& eval;

};

#endif
