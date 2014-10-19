/*
 <moSwapNeighbor.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#ifndef _moSwapNeighbor_h
#define _moSwapNeighbor_h

#include "../../neighborhood/moBackableNeighbor.h"

/**
 * Swap Neighbor
 */
template<class EOT, class Fitness = typename EOT::Fitness>
class moSwapNeighbor: public moBackableNeighbor<EOT, Fitness> {
public:

	/**
	 * Apply the swap
	 * @param _solution the solution to move
	 */
	virtual void move(EOT& _solution) {
		EOT tmp(1);
		tmp[0] = _solution[indices.first];
		_solution[indices.first] = _solution[indices.second];
		_solution[indices.second] = tmp[0];
		_solution.invalidate();
	}

	/**
	 * apply the swap to restore the solution (use by moFullEvalByModif)
	 * @param _solution the solution to move back
	 */
	virtual void moveBack(EOT& _solution) {
		move(_solution);
	}

	/**
	 * Setter to fix the two indexes to swap
	 * @param _first first index
	 * @param _second second index
	 */
	void setIndices(unsigned int _first, unsigned int _second) {
		indices.first = _first;
		indices.second = _second;
	}

	/**
	 * Get the two indexes to swap
	 * @param _first first index
	 * @param _second second index
	 */
	void getIndices(unsigned int & _first, unsigned int & _second) {
		_first = indices.first;
		_second = indices.second;
	}

	/**
	 * @param _neighbor a neighbor
	 * @return if _neighbor and this one are equals
	 */
	virtual bool equals(moSwapNeighbor<EOT,Fitness>& _neighbor) {
		unsigned f, s;
		_neighbor.getIndices(f, s);
		return (((indices.first == f) && (indices.second == s)) || ((indices.first == s) && (indices.second == f)));
	}

	/**
	 * Print the Neighbor
	 */
	void print() {
		std::cout << "[" << indices.first << ", " << indices.second << "] -> "
				<< (*this).fitness() << std::endl;
	}

private:
	std::pair<unsigned int, unsigned int> indices;

};

#endif
