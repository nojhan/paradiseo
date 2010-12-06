/*
 <moKSwapNeighbor.h>
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

#ifndef _moKswapNeighbor_h
#define _moKswapNeighbor_h

#include <neighborhood/moBackableNeighbor.h>
#include <neighborhood/moIndexSwapNeighbor.h>

/**
 * K Swap Neighbor
 */

template<class EOT, class Fitness = typename EOT::Fitness>
class moKswapNeighbor: public moBackableNeighbor<EOT> ,
		public moIndexSwapNeighbor<EOT> {
public:
	using moIndexSwapNeighbor<EOT>::indices;
	using moIndexSwapNeighbor<EOT>::Kswap;
	using moIndexSwapNeighbor<EOT>::size;

	/**
	 *Default Constructor
	 */

	moKswapNeighbor() :
		moIndexSwapNeighbor<EOT>() {
	}

	/**
	 * Constructor
	 * @param _Kswap the number of swap to do
	 */

	moKswapNeighbor() :
		moIndexSwapNeighbor<EOT>(Kswap) {

	}

	/**
	 * Apply the K-swap
	 * @param _solution the solution to move
	 */
	virtual void move(EOT& _solution) {
		size = _solution.size();
		EOT tmp(1);
		for (unsigned int i = 0; i < Kswap; i++) {
			tmp[0] = _solution[indices[i]];
			_solution[indices[i]] = _solution[indices[i + 1]];
			_solution[indices[i + 1]] = tmp[0];
		}
		_solution.invalidate();
	}

	/**
	 * apply the K-swap to restore the solution (use by moFullEvalByModif)
	 * @param _solution the solution to move back
	 */
	virtual void moveBack(EOT& _solution) {
		EOT tmp(1);
		for (int i = Kswap; i > 0; i--) {
			tmp[0] = _solution[indices[i]];
			_solution[indices[i]] = _solution[indices[i - 1]];
			_solution[indices[i - 1]] = tmp[0];
		}
		_solution.invalidate();
	}

	/**
	 * Return the class name.
	 * @return the class name as a std::string
	 */
	virtual std::string className() const {
		return "moKswapNeighbor";
	}

};

#endif

