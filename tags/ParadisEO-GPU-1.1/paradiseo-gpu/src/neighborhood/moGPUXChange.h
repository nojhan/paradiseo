/*
 <moGPUXChange.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

 Karima  Boufaras, Thé Van Luong

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

#ifndef _moGPUXChange_h
#define _moGPUXChange_h

#include <neighborhood/moGPUMapping.h>

/**
 * Generalization of exchange and hamming distance based Neighborhood
 */

template<class N>
class moGPUXChange: public moGPUMapping<N> {

public:

	/**
	 * Define type of a solution corresponding to Neighbor
	 */
	typedef N Neighbor;
	typedef typename Neighbor::EOT EOT;

	using moGPUMapping<Neighbor>::neighborhoodSize;
	using moGPUMapping<Neighbor>::currentIndex;
	using moGPUMapping<Neighbor>::indices;
	using moGPUMapping<Neighbor>::mapping;
	using moGPUMapping<Neighbor>::xChange;
	using moGPUMapping<Neighbor>::mutex;

	/**
	 * Constructor
	 * @param _neighborhoodSize the neighborhood size
	 * @param _xChange the number of x-change positions
	 */

	moGPUXChange(unsigned int _neighborhoodSize, unsigned int _xChange) :
		moGPUMapping<Neighbor> (_neighborhoodSize, _xChange) {
	}


	/**
	 * Set the mapping of K-indexes
	 * @param _size the solution size
	 */

	void setMapping(unsigned _size) {

		for (unsigned int i = 0; i < xChange; i++)
			indices[i] = i;

		unsigned int id = 0;
		bool change = false;

		while (id < neighborhoodSize) {

			while (indices[xChange - 1] < _size) {

				updateMapping(indices, id);
				indices[(xChange - 1)]++;
				id++;
			}

			indices[xChange - 1]--;

			if (id < neighborhoodSize) {
				for (int i = xChange - 2; i >= 0; i--)
					if (!change)
						change = nextIndices(i, _size);
			}

			change = false;
		}

	}

	/**
	 * Compute the next  combination of mapping  indices
	 * @param _indice  compute next combination of indexes  from this index
	 * @param _size the solution size
	 */

	bool nextIndices(int _indice, unsigned _size) {

		if (indices[_indice + 1] == _size - xChange + 1 + _indice) {

			if (indices[_indice] + 1 < _size - xChange + 1 + _indice) {

				indices[_indice]++;
				unsigned int i = 1;

				while (_indice + i < xChange) {

					indices[_indice + i] = indices[_indice + i - 1] + 1;
					i++;

				}

				return true;

			} else {

				return false;
			}

		} else {

			indices[_indice + 1]++;
			unsigned int i = 2;

			while (_indice + i < xChange) {

				indices[_indice + i] = indices[_indice + i - 1] + 1;
				i++;

			}

			return true;
		}

	}

	/**
	 * Setter to fix the number of x-change positions
	 * @param _xChange the number of x-change
	 */

	void setXChange(unsigned int _xChange) {
		xChange = _xChange;
	}

	/**
	 * Get the number of x-change
	 */

	unsigned int getXChange() {
		return xChange;
	}


	/**
	 * Return the class Name
	 * @return the class name as a std::string
	 */
	virtual std::string className() const {
		return "moGPUXChange";
	}

};

#endif
