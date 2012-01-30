/*
 <moGPUXBitFlippingN.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

 Karima Boufaras , Thé Van Luong

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

#ifndef _moGPUXBitFlippingN_h
#define _moGPUXBitFlippingN_h

#include <neighborhood/moBackableNeighbor.h>
#include <neighborhood/moXChangeNeighbor.h>

/**
 * X-BitFlipping Neighbor
 * Neighbor represent different hamming distances,it's fixed by xFlip attribute
 */

template<class EOT, class Fitness = typename EOT::Fitness>
class moGPUXBitFlippingN:public moBackableNeighbor<EOT> ,
		public moXChangeNeighbor<EOT> {
public:

	using moXChangeNeighbor<EOT>::indices;
	using moXChangeNeighbor<EOT>::xChange;
	using moXChangeNeighbor<EOT>::key;

	/**
	 *Default Constructor
	 */

	moGPUXBitFlippingN() :
		moXChangeNeighbor<EOT> () {
	}

	/**
	 * Constructor
	 * @param _xFlip the number of bit to flip
	 */

	moGPUXBitFlippingN(unsigned int _xFlip) :
		moXChangeNeighbor<EOT> (_xFlip) {
	}

	/**
	 * Apply the K-Flip in solution
	 * @param _solution the solution to move
	 */

	virtual void move(EOT& _solution) {
			for (unsigned int i = 0; i < xChange; i++)
				_solution[indices[i]] = !_solution[indices[i]];
		_solution.invalidate();

	}

	/**
	 * apply the K-Flip to restore the solution (use by moFullEvalByModif)
	 * @param _solution the solution to move back
	 */

	virtual void moveBack(EOT& _solution) {
		move(_solution);
	}

	/**
	 * Return the class name.
	 * @return the class name as a std::string
	 */

	virtual std::string className() const {
		return "moGPUXBitFlippingN";
	}

};

#endif

