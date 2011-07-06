/*
 <moMappingNeighborhood.h>
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

#ifndef _moMappingNeighborhood_h
#define _moMappingNeighborhood_h

#include <neighborhood/moOrderNeighborhood.h>

/**
 * Define a Mapping Neighborhood:
 *
 * This neighborhood should manipulate a moXChangeNeighbor
 * It helps to associate to identified neighbor  by one index a set of correspondent indexes
 */

template<class N>
class moMappingNeighborhood: public moOrderNeighborhood<N> {

public:

	/**
	 * Define type of a solution corresponding to Neighbor
	 */

	typedef N Neighbor;
	typedef typename Neighbor::EOT EOT;

	using moOrderNeighborhood<Neighbor>::neighborhoodSize;
	using moOrderNeighborhood<Neighbor>::currentIndex;

	/**
	 * @Constructor
	 * @param _neighborhoodSize the neighborhood size
	 * @param _xChange the number of positions to exchange
	 */

	moMappingNeighborhood(unsigned int _neighborhoodSize, unsigned int _xChange) :
		moOrderNeighborhood<Neighbor> (_neighborhoodSize), xChange(_xChange) {

		mutex = false;
		indices = new unsigned int[xChange];
		mapping = new unsigned int[neighborhoodSize * xChange];

	}

	/**
	 * Destructor
	 */

	~moMappingNeighborhood() {

		delete[] (indices);
		delete[] (mapping);
	}

	/**
	 * Initialization of the neighborhood
	 * @param _solution the solution to explore
	 * @param _current the first neighbor
	 */

	virtual void init(EOT& _solution, Neighbor& _current) {

		moOrderNeighborhood<Neighbor>::init(_solution, _current);
		//Compute the mapping only for the first init
		if (!mutex) {
			setMapping(_solution.size());
			mutex = true;
		}
		//get the mapping correspondent to the currentIndex
		getMapping(currentIndex);
		if (!(_current.getXChange() == xChange))
			_current.setXChange(xChange);
		//associate to this neighbor a set of correspondent indexes
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
	 * Set the mapping of K-indexes
	 * @param _size the size of the solution
	 */

	virtual void setMapping(unsigned _size) =0;

	/**
	 * Associate mapping of current index to a set of indexes
	 * _currentIndex the index to map with a set of indexes
	 */

	virtual void getMapping(unsigned int _currentIndex) {
		for (unsigned int i = 0; i < xChange; i++) {
			indices[i] = mapping[_currentIndex + i * neighborhoodSize];
		}
	}

	/**
	 * update mapping with combination of indexes corresponding to the currentIndex
	 * @param _indices the set of current combination of indexes
	 * @param _currentIndex  the current index corresponding to the current neighbor
	 */

	virtual void updateMapping(unsigned int* _indices, int _currentIndex) {
		for (unsigned k = 0; k < xChange; k++)
			mapping[_currentIndex + k * neighborhoodSize] = _indices[k];
	}

	/**
	 * Return the class Name
	 * @return the class name as a std::string
	 */
	virtual std::string className() const {
		return "moMappingNeighborhood";
	}

protected:

	unsigned int * indices;
	unsigned int * mapping;
	unsigned int xChange;
	bool mutex;
};

#endif
