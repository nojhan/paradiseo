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

#ifndef _moKSwapNeighborhood_h
#define _moKSwapNeighborhood_h

#include <neighborhood/moOrderNeighborhood.h>

/**
 * @return the factorial of an integer
 * @param _i an integer
 */

static int factorial(int _i) {
	if (_i == 0)
		return 1;
	else
		return _i * factorial(_i - 1);
}

/**
 * @return the neighborhood Size from the solution size and number of swap
 * @param _size the solution size
 * @param _Kswap the number of swap
 */

static int sizeMapping(int _size,unsigned int _Kswap) {
	int _sizeMapping = _size;
	for (int i = _Kswap; i > 0; i--) {
		_sizeMapping *= (_size - i);
	}
	_sizeMapping /= factorial(_Kswap + 1);
	return _sizeMapping;
}

/**
 * K-Swap Neighborhood
 */

template<class N>
class moKswapNeighborhood: public moOrderNeighborhood<N> {

public:

	typedef N Neighbor;
	typedef typename Neighbor::EOT EOT;

	using moOrderNeighborhood<Neighbor>::neighborhoodSize;
	using moOrderNeighborhood<Neighbor>::currentIndex;

	/**
	 * @Constructor
	 * @param _size the solution size
	 * @param _Kswap the number of swap
	 */

	moKswapNeighborhood(unsigned int _size, unsigned int _Kswap) :
		moOrderNeighborhood<Neighbor> (sizeMapping(_size, _Kswap)) {

		mutex = true;
		size = _size;
		Kswap = _Kswap;
		indices = new unsigned int[Kswap + 1];
		mapping = new unsigned int[neighborhoodSize * (Kswap + 1)];

	}
	;

	/**
	 * Destructor
	 */

	~moKswapNeighborhood() {
		delete[] (indices);
		delete[] (mapping);
	}

	/**
	 * Initialization of the neighborhood
	 * @param _solution the solution to explore
	 * @param _current the first neighbor
	 */
	virtual void init(EOT& _solution, Neighbor& _current) {

		//Compute the mapping only for the first init
		if (mutex==true) {
			setMapping();
			mutex = false;
		}
		moOrderNeighborhood<Neighbor>::init(_solution, _current);
		_current.setKswap(Kswap);
		_current.reSizeIndices(Kswap);
		_current.setSize(_solution.size());
		getMapping(currentIndex);
		_current.setIndices(indices);
	}

	/**
	 * Give the next neighbor
	 * @param _solution the solution to explore
	 * @param _current the next neighbor
	 */
	virtual void next(EOT& _solution, Neighbor& _current) {
		moOrderNeighborhood<Neighbor>::next(_solution, _current);
		getMapping(currentIndex);
		_current.setIndices(indices);
	}

	/**
	 * Associate mapping of current index to K indexes
	 * _currentIndex the index to map with K indexes
	 */

	virtual void getMapping(unsigned int _currentIndex) {
		for (unsigned int i = 0; i <= Kswap; i++){
			indices[i] = mapping[_currentIndex + i * neighborhoodSize];
		}
	}

	/**
	 * Set the mapping of K-indexes
	 * @param _size the solution size
	 */

	void setMapping() {

		if (!Kswap) {
			for (unsigned int i = 0; i < size; i++)
				mapping[i] = i;
		} else {
			unsigned int * _indices;

			_indices = new unsigned int[Kswap + 1];

			for (unsigned int i = 0; i < Kswap + 1; i++)
				_indices[i] = i;

			unsigned int id = 0;
			bool change = false;
			while (id < neighborhoodSize) {

				while (_indices[Kswap] < size) {

					for (unsigned int k = 0; k < Kswap; k++) {
						mapping[id + k * neighborhoodSize] = _indices[k];
					}

					mapping[id + Kswap * neighborhoodSize] = _indices[Kswap]++;
					id++;
				}

				_indices[Kswap]--;

				if (id < neighborhoodSize) {
					for (int i = Kswap - 1; i >= 0; i--)
						if (!change)
							change = nextIndices(_indices, i);
				}

				change = false;
			}

			delete[] (_indices);
		}
	}

	/**
	 * Compute the next  combination of mapping  indexes
	 * @param _indices the current combination of indexes
	 * @param _indice  compute next combination of indexes  from this index
	 */
	bool nextIndices(unsigned int* _indices, int _indice) {

		if (_indices[_indice + 1] == size - Kswap + _indice) {

			if (_indices[_indice] + 1 < size - Kswap + _indice) {

				_indices[_indice]++;
				unsigned int i = 1;

				while (_indice + i <= Kswap) {

					_indices[_indice + i] = _indices[_indice + i - 1] + 1;
					i++;
				}

				return true;

			} else {

				_indices[_indice] = _indices[_indice];
				return false;
			}

		} else {

			_indices[_indice + 1]++;
			unsigned int i = 2;

			while (_indice + i <= Kswap) {
				_indices[_indice + i] = _indices[_indice + i - 1] + 1;
				i++;
			}

			return true;
		}

	}

	/**
	 * Setter to fix the Kswap
	 * @param _Kswap the number of swap
	 */

	void setKswap(unsigned int _Kswap) {
		Kswap = _Kswap;
	}

	/**
	 * Get the number of swap
	 */

	unsigned int getKswap() {
		return Kswap;
	}

	/**
	 * Setter to fix the neighbor size
	 * @param _size the neighbor size to set
	 */

	void setSize(unsigned int _size) {
		size = _size;
	}

	/**
	 * Get the size of neighbor
	 */

	unsigned int getSize() {
		return size;
	}

	/**
	 * Return the class Name
	 * @return the class name as a std::string
	 */
	virtual std::string className() const {
		return "moKSwapNeighborhood";
	}

protected:

	unsigned int * indices;
	unsigned int * mapping;
	unsigned int Kswap;
	unsigned int size;
	bool mutex;
};

#endif
